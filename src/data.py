from __future__ import annotations

import json
import logging
from functools import partial
from pathlib import Path

import torch
import torchaudio
import transformers
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm import tqdm

from utils import mask2lens


class SQADataset(Dataset):
    def __init__(self, datasets: list[Path] | Path, metrics: list[str], sample_rate: int = 16000):
        self.entries = []
        self.sample_rate = sample_rate
        self.metrics = metrics

        if isinstance(datasets, Path):
            datasets = [datasets]

        for dataset in datasets:
            logging.info(f"Loading dataset: {dataset}")
            with open(dataset, "r") as f:
                for line in tqdm(f):
                    self.entries.append(json.loads(line))

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        item = self.entries[idx]
        audio, sr = torchaudio.load(item["audio_path"])
        if sr != self.sample_rate:
            audio = torchaudio.functional.resample(audio, sr, self.sample_rate)
        audio = audio.mean(dim=0)  # mono
        item_dict = {
            "audio": audio,
            "sample_id": item["sample_id"],
            "system_id": item["system_id"],
        }
        if "metrics" in item:
            item_dict["metrics"] = {name: item["metrics"].get(name, float("nan")) for name in self.metrics}
        return item_dict

    def get_frame_len(self, idx):
        item = self.entries[idx]
        try:
            duration = item["duration"]
        except KeyError:
            info = torchaudio.info(item["audio_path"])
            duration = info.num_frames / info.sample_rate
        return int(duration * self.sample_rate)


def collate_fn(batch, feature_extractor: transformers.SequenceFeatureExtractor = None):
    audio_list = [item["audio"] for item in batch]
    sample_ids = [item["sample_id"] for item in batch]
    system_ids = [item["system_id"] for item in batch]
    if feature_extractor:
        inputs = feature_extractor([audio.numpy() for audio in audio_list], return_tensors="pt", padding=True)
        audio = inputs.input_values
        audio_lengths = mask2lens(inputs.attention_mask)
    else:
        audio = torch.nn.utils.rnn.pad_sequence(audio_list, batch_first=True)
        audio_lengths = torch.tensor([len(audio) for audio in audio_list], dtype=torch.long)

    batch_dict = {
        "audio": audio,
        "audio_lengths": audio_lengths,
        "sample_ids": sample_ids,
        "system_ids": system_ids,
    }
    if "metrics" in batch[0]:
        metrics_list: list[dict[str, float]] = [item["metrics"] for item in batch]
        metrics: dict[str, float["b"]] = {
            m: torch.tensor([item[m] for item in metrics_list], dtype=torch.float32) for m in metrics_list[0].keys()
        }
        batch_dict["metrics"] = metrics
    return batch_dict


# https://github.com/SWivid/F5-TTS/blob/605fa13b42b40e860961bac8ce30fe49f02dfa0d/src/f5_tts/model/dataset.py#L165
class DynamicBatchSampler(Sampler):
    """Extension of Sampler that will do the following:
    1.  Change the batch size (essentially number of sequences)
        in a batch to ensure that the total number of frames are less
        than a certain threshold.
    2.  Make sure the padding efficiency in the batch is high.
    3.  Shuffle batches each epoch while maintaining reproducibility.
    """

    def __init__(
        self, dataset: Dataset, frames_threshold: int, max_samples=0, random_seed=None, drop_residual: bool = False
    ):
        self.frames_threshold = frames_threshold
        self.max_samples = max_samples
        self.random_seed = random_seed
        self.epoch = 0

        indices, batches = [], []
        logging.info("Sorting dataset by frame lengths... This can be slow if duration was not precomputed")
        for idx in tqdm(range(len(dataset)), desc="Sorting dataset... "):
            indices.append((idx, dataset.get_frame_len(idx)))
        indices.sort(key=lambda elem: elem[1])

        batch = []
        batch_frames = 0
        for idx, frame_len in tqdm(
            indices, desc=f"Creating dynamic batches with {frames_threshold} audio frames per gpu"
        ):
            if batch_frames + frame_len <= self.frames_threshold and (max_samples == 0 or len(batch) < max_samples):
                batch.append(idx)
                batch_frames += frame_len
            else:
                if len(batch) > 0:
                    batches.append(batch)
                if frame_len <= self.frames_threshold:
                    batch = [idx]
                    batch_frames = frame_len
                else:
                    logging.warning(
                        f"Single sample with {frame_len} frames exceeds the frames_threshold of {self.frames_threshold}, dropping it."
                    )
                    batch = []
                    batch_frames = 0

        if not drop_residual and len(batch) > 0:
            batches.append(batch)

        del indices
        self.batches = batches

        # Ensure even batches with accelerate BatchSamplerShard cls under frame_per_batch setting
        self.drop_last = True

    def set_epoch(self, epoch: int) -> None:
        """Sets the epoch for this sampler."""
        self.epoch = epoch

    def __iter__(self):
        # Use both random_seed and epoch for deterministic but different shuffling per epoch
        if self.random_seed is not None:
            g = torch.Generator()
            g.manual_seed(self.random_seed + self.epoch)
            # Use PyTorch's random permutation for better reproducibility across PyTorch versions
            indices = torch.randperm(len(self.batches), generator=g).tolist()
            batches = [self.batches[i] for i in indices]
        else:
            batches = self.batches
        return iter(batches)

    def __len__(self):
        return len(self.batches)


def init_dataloader(
    data: list[Path],
    metrics: dict[str, dict[str, float]],
    sample_rate: int = 16000,
    batch_frame_per_gpu: int = 480000,
    max_samples_per_gpu: int = 32,
    num_workers: int = 4,
    prefetch: int = 100,
    feature_extractor: transformers.SequenceFeatureExtractor = None,
    seed: int = 42,
    is_train: bool = True,
) -> DataLoader:

    dataset = SQADataset(data, metrics, sample_rate)
    batch_sampler = DynamicBatchSampler(
        dataset,
        batch_frame_per_gpu,
        max_samples=max_samples_per_gpu,
        random_seed=seed if is_train else None,
    )
    if feature_extractor:
        feature_extractor_partial = partial(feature_extractor, sampling_rate=sample_rate)
        collate_fn_partial = partial(collate_fn, feature_extractor=feature_extractor_partial)
    else:
        collate_fn_partial = collate_fn

    dataloader = DataLoader(
        dataset,
        collate_fn=collate_fn_partial,
        num_workers=num_workers,
        prefetch_factor=prefetch,
        pin_memory=num_workers != 0,
        persistent_workers=num_workers != 0,
        batch_sampler=batch_sampler,
    )
    return dataloader
