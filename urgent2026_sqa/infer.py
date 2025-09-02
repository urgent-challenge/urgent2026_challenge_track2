import argparse
import json
import logging
from functools import partial
from pathlib import Path
from typing import Optional

import torch
import torchaudio
from huggingface_hub import snapshot_download
from hyperpyyaml import load_hyperpyyaml
from tqdm import tqdm

from urgent2026_sqa.data import init_dataloader
from urgent2026_sqa.utils import mask2lens, override


def load_model(checkpoint: Path, config: Optional[dict] = None):
    if not checkpoint.is_file():
        logging.info(f"{checkpoint} is not a file, assume it's a huggingface hub repo id")
        checkpoint = Path(snapshot_download(checkpoint.as_posix())) / "model.pt"

    if config is None:
        config_path = checkpoint.parent / "config.yaml"
        if not config_path.is_file():
            raise ValueError("failed to find config.yaml")
    with open(config_path) as f:
        config = load_hyperpyyaml(f)

    state_dict = torch.load(checkpoint, map_location="cpu")["model"]

    model = config["model"]
    model.load_state_dict(state_dict)
    model.eval()
    return model, config


@torch.inference_mode()
def infer(model, config: dict, data: Path, outdir: Path):
    from accelerate import Accelerator

    accelerator = Accelerator()
    config["dataloader"] = override(
        config["dataloader"],
        num_workers=args.num_workers,
        prefetch=args.prefetch,
        batch_frame_per_gpu=args.batch_frames,
    )
    dataloader = init_dataloader(data, **config["dataloader"], is_train=False)
    model, dataloader = accelerator.prepare(model, dataloader)
    results = []
    with open(outdir / "results.jsonl", "w") as f:
        for batch in tqdm(dataloader):
            batch_metric2preds = model.predict(**batch)
            for i, (sample_id, system_id) in enumerate(zip(batch["sample_ids"], batch["system_ids"])):
                item = {
                    "sample_id": sample_id,
                    "system_id": system_id,
                    "metrics": {k: float(batch_metric2preds[k][i]) for k in batch_metric2preds},
                }
                results.append(item)
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    for metric in model.metrics:
        with open(outdir / f"{metric}.scp", "w") as f:
            for item in results:
                f.write(f"{item['sample_id']} {item['metrics'][metric]}\n")


@torch.inference_mode()
def infer_single(model, config: dict, audio: torch.Tensor | Path, audio_sr: int = None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    if isinstance(audio, Path):
        audio, audio_sr = torchaudio.load(audio.as_posix())
    elif isinstance(audio, torch.Tensor):
        assert audio_sr is not None, "Please provide audio sample rate"
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
    parser.add_argument("--num_workers", default=4, type=int, help="num of subprocess workers for reading")
    parser.add_argument("--prefetch", default=100, type=int, help="prefetch number")
    parser.add_argument("--batch-frames", default=480000, type=int, help="batch frames for inference")
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
        infer(model, config, args.data, args.outdir)
        logging.info(f"Inference results are saved in {args.outdir}")
