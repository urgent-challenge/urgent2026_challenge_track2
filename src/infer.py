import argparse
import json
import os
import tempfile
from pathlib import Path

import torch
from accelerate import Accelerator
from hyperpyyaml import load_hyperpyyaml
from tqdm import tqdm

from data import init_dataloader
from utils import override


@torch.inference_mode()
def infer(model, dataloader, outdir):
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


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, type=Path, help="path to checkpoint for inference")
    parser.add_argument("--data", required=True, type=Path, help="path to jsonl file or audio file")
    parser.add_argument("--outdir", required=True, type=Path, help="path to output dir")
    parser.add_argument("--config", type=Path, help="path to config file yaml")
    parser.add_argument("--num_workers", default=4, type=int, help="num of subprocess workers for reading")
    parser.add_argument("--prefetch", default=100, type=int, help="prefetch number")
    parser.add_argument("--batch-frames", default=480000, type=int, help="batch frames for inference")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()

    if args.config is None:
        args.config = args.ckpt.parent / "config.yaml"
        if not args.config.is_file():
            raise ValueError("No config file is found for the checkpoint, please specify with --config")

    if args.outdir is None:
        args.outdir = args.ckpt.parent / "infer" / args.data.stem
    args.outdir.mkdir(parents=True, exist_ok=True)

    with open(args.config) as f:
        config = load_hyperpyyaml(f)

    config["dataloader"] = override(
        config["dataloader"],
        num_workers=args.num_workers,
        prefetch=args.prefetch,
        batch_frame_per_gpu=args.batch_frames,
    )
    if args.data.suffix == ".jsonl":
        dataloader = init_dataloader(args.data, **config["dataloader"], is_train=False)
    else:
        item_dict = {"audio_path": args.data.as_posix(), "sample_id": args.data.stem, "system_id": args.data.stem}
        fd, path = tempfile.mkstemp(suffix=".jsonl")
        with os.fdopen(fd, "w") as f:
            f.write(json.dumps(item_dict, ensure_ascii=False))
        dataloader = init_dataloader(Path(path), **config["dataloader"], is_train=False)

    state_dict = torch.load(args.ckpt, map_location="cpu")["model_state_dict"]
    model = config["model"]
    model.load_state_dict(state_dict)

    accelerator = Accelerator()
    model, dataloader = accelerator.prepare(model, dataloader)

    infer(model, dataloader, args.outdir)
    print(f"Inference results are saved in {args.outdir}")
