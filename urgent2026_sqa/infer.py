from __future__ import annotations

import argparse
import json
import logging
import os
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torchaudio
from accelerate import Accelerator
from huggingface_hub import hf_hub_download
from hyperpyyaml import load_hyperpyyaml
from tqdm import tqdm

from urgent2026_sqa.data import init_dataloader
from urgent2026_sqa.utils import mask2lens, override


def load_model(checkpoint: Path | str, config: str | Path = None):
    if not os.path.isfile(checkpoint):
        logging.info(f"{checkpoint} is not a file, assume it's a huggingface hub repo id")
        checkpoint, config = hf_hub_download(checkpoint, "model.pt"), hf_hub_download(checkpoint, "config.yaml")
    checkpoint = Path(checkpoint)

    if config is None:
        config = checkpoint.parent / "config.yaml"
        if not config.is_file():
            raise ValueError("failed to find config.yaml")
    with open(config) as f:
        config = load_hyperpyyaml(f)

    state_dict = torch.load(checkpoint, map_location="cpu")["model"]

    model = config["model"]
    model.load_state_dict(state_dict)
    model.eval()
    return model, config


@torch.inference_mode()
def infer(
    model,
    config: dict,
    data: Path | str | list[dict] | list,
    num_workers=1,
    prefetch=10,
    batch_frames=4_800_000,
):
    accelerator = Accelerator()
    config["dataloader"] = override(
        config["dataloader"],
        num_workers=num_workers,
        prefetch=prefetch,
        batch_frame_per_gpu=batch_frames,
    )
    dataloader = init_dataloader(data, **config["dataloader"], is_train=False)
    model, dataloader = accelerator.prepare(model, dataloader)
    results = []
    for batch in tqdm(dataloader):
        batch_metric2preds = model.predict(**batch)
        for i, (sample_id, system_id) in enumerate(zip(batch["sample_ids"], batch["system_ids"])):
            item = {
                "sample_id": sample_id,
                "system_id": system_id,
                "metrics": {k: float(batch_metric2preds[k][i]) for k in batch_metric2preds},
            }
            results.append(item)
    return results


@torch.inference_mode()
def infer_single(model, config: dict, audio: torch.Tensor | np.ndarray | Path | str, audio_sr: int = None):
    """
    when audio is torch.Tensor or np.array, the shap shoud be (1, num_samples)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    if isinstance(audio, (Path, str)):
        audio = Path(audio)
        audio, audio_sr = torchaudio.load(audio.as_posix())
    elif isinstance(audio, (torch.Tensor, np.ndarray)):
        assert audio_sr is not None, "Please provide audio sample rate"
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio)
        assert audio.dim() == 2, "audio should be (1, num_samples)"
    else:
        raise ValueError("audio should be a torch.Tensor or Path")
    sample_rate = config["dataloader"].get("sample_rate", 16000)
    if audio_sr != sample_rate:
        audio = torchaudio.functional.resample(audio, audio_sr, feature_extractor.sampling_rate)

    audio = audio.mean(dim=0).unsqueeze(0)  # mono
    feature_extractor = config["dataloader"].get("feature_extractor", None)
    if feature_extractor is not None:
        feature_extractor_partial = partial(feature_extractor, sampling_rate=sample_rate)
        inputs = feature_extractor_partial(audio.numpy(), sampling_rate=sample_rate, return_tensors="pt")
        audio, audio_lengths = inputs.input_values, mask2lens(inputs.attention_mask)
    else:
        audio_lengths = torch.tensor([audio.shape[1]], dtype=torch.long)
    audio, audio_lengths = audio.to(device), audio_lengths.to(device)
    inputs = {"audio": audio, "audio_lengths": audio_lengths}
    metric2preds = {k: float(v[0]) for k, v in model.predict(**inputs).items()}
    return metric2preds


@torch.inference_mode()
def infer_list(model, config: dict, audios: list[Path | str]):
    data = [{"sample_id": i, "system_id": "default", "audio_path": audio} for i, audio in enumerate(audios)]
    for item in data:
        info = torchaudio.info(item["audio_path"].as_posix())
        item["duration"] = info.num_frames / info.sample_rate
    results = infer(model, config, data)
    results.sort(key=lambda x: x["sample_id"])
    return [item["metrics"] for item in results]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        required=True,
        type=Path,
        default="vvwang/urgent2026-uni_ver_ext-5_metrics",
        help="path to checkpoint or huggingface repo id",
    )
    parser.add_argument("--data", required=True, type=Path, help="path to jsonl file or audio file")
    parser.add_argument("--outdir", type=Path, help="path to output dir")
    parser.add_argument("--config", type=Path, help="path to config file yaml")
    parser.add_argument("--num-workers", default=4, type=int, help="num of subprocess workers for reading")
    parser.add_argument("--prefetch", default=100, type=int, help="prefetch number")
    parser.add_argument("--batch-frames", default=4_800_000, type=int, help="batch frames for inference")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    model, config = load_model(args.ckpt, args.config)
    if args.data.suffix not in [".jsonl", ".scp"]:
        metric2preds = infer_single(model, config, args.data)
        print(json.dumps(metric2preds, ensure_ascii=False, indent=4))
    else:
        assert args.outdir is not None, "Please provide --outdir for batch inference"
        args.outdir.mkdir(parents=True, exist_ok=True)

        results = infer(model, config, args.data, args.num_workers, args.prefetch, args.batch_frames)
        with open(args.outdir / "results.jsonl", "w") as f:
            for item in results:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        for metric in model.metrics:
            with open(args.outdir / f"{metric}.scp", "w") as f:
                for item in results:
                    f.write(f"{item['sample_id']} {item['metrics'][metric]}\n")

        logging.info(f"Inference results are saved in {args.outdir}")
