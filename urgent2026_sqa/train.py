#!/usr/bin/env python3
import argparse
import shutil
from pathlib import Path

from hyperpyyaml import load_hyperpyyaml
from trainer import Trainer

from urgent2026_sqa.data import init_dataloader
from urgent2026_sqa.utils import override


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path, help="path to config file yaml")
    parser.add_argument("--train-data", required=True, type=Path, nargs="+", help="path to jsonl files for training")
    parser.add_argument("--cv-data", required=True, type=Path, nargs="+", help="path to jsonl files for vadidation")
    parser.add_argument("--exp", required=True, type=Path, default=None)
    parser.add_argument(
        "--pretrained-ckpt", type=Path, help="path to model checkpoint to initialize from", default=None
    )
    parser.add_argument("--num-workers", default=4, type=int, help="num of subprocess workers for reading")
    parser.add_argument("--prefetch", default=100, type=int, help="prefetch number")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()

    with open(args.config) as f:
        config = load_hyperpyyaml(f)

    args.exp.mkdir(parents=True, exist_ok=True)

    shutil.copy(args.config, args.exp / "config.yaml")
    config["dataloader"] = override(config["dataloader"], num_workers=args.num_workers, prefetch=args.prefetch)
    train_dataloader = init_dataloader(args.train_data, **config["dataloader"], is_train=True)
    cv_dataloader = init_dataloader(args.cv_data, **config["dataloader"], is_train=False)

    trainer = Trainer(
        model=config["model"],
        exp_dir=args.exp,
        pretrained_ckpt=args.pretrained_ckpt,
        **config["trainer"],
    )
    trainer.train(train_dataloader, cv_dataloader, seed=config["seed"])
