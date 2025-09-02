from __future__ import annotations

import logging
import math
from pathlib import Path

import torch
import torch.distributed as dist
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from ema_pytorch import EMA
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from utils import calculate_metrics

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        exp_dir: Path,
        pretrained_ckpt: Path = None,
        epochs: int = 100,
        learning_rate: float = 1e-4,
        num_warmup_updates: int = 20000,
        grad_accumulation_steps=1,
        grad_norm=1.0,
        save_per_updates=1000,
        keep_last_n_checkpoints=-1,
        ema_kwargs: dict = dict(),
        accelerate_kwargs: dict = dict(),
    ):
        self.pretrained_ckpt = pretrained_ckpt
        self.exp_dir = exp_dir

        self.model = model

        self.epochs = epochs
        self.num_warmup_updates = num_warmup_updates
        self.save_per_updates = save_per_updates
        self.keep_last_n_checkpoints = keep_last_n_checkpoints

        self.learning_rate = learning_rate
        self.grad_accumulation_steps = grad_accumulation_steps
        self.grad_norm = grad_norm

        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        try:
            import tensorboard
        except ImportError:
            raise ImportError("TensorBoard is not installed. Please run `pip install tensorboard`.")
        self.accelerator = Accelerator(
            log_with="tensorboard",
            project_dir="./runs",
            kwargs_handlers=[ddp_kwargs],
            gradient_accumulation_steps=self.grad_accumulation_steps,
            **accelerate_kwargs,
        )
        model_cfg_dict = {
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "num_warmup_updates": self.num_warmup_updates,
            "grad_accumulation_steps": self.grad_accumulation_steps,
            "grad_norm": self.grad_norm,
            "gpus": self.accelerator.num_processes,
        }
        self.accelerator.init_trackers(
            project_name=self.exp_dir.name,
            config=model_cfg_dict,
        )
        if self.is_main:
            self.ema_model = EMA(self.model, include_online_model=False, **ema_kwargs)
            self.ema_model.to(self.accelerator.device)

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    def save_checkpoint(self, update, is_last=False):
        self.accelerator.wait_for_everyone()
        if not self.is_main:
            return

        checkpoint = dict(
            model=self.accelerator.unwrap_model(self.model).state_dict(),
            optimizer=self.optimizer.state_dict(),
            ema_model=self.ema_model.state_dict(),
            scheduler=self.scheduler.state_dict(),
            update=update,
        )
        if is_last:
            self.accelerator.save(checkpoint, self.exp_dir / "model_last.pt")
            logger.info(f"Saved last checkpoint at update {update}")
            return

        if self.keep_last_n_checkpoints == 0:
            return

        self.accelerator.save(checkpoint, f"{self.exp_dir}/model_{update}.pt")
        if self.keep_last_n_checkpoints > 0:
            checkpoints = [ckpt for ckpt in self.exp_dir.glob("*.pt") if ckpt.stem != "model_last"]
            checkpoints.sort(key=lambda p: int(p.stem.removeprefix("model_")))
            while len(checkpoints) > self.keep_last_n_checkpoints:
                earliest_checkpoint = checkpoints.pop(0)
                earliest_checkpoint.unlink()
                logger.info(f"Removed early checkpoint: {earliest_checkpoint}")

    def load_checkpoint(self):
        self.accelerator.wait_for_everyone()

        latest_checkpoint = None
        if self.pretrained_ckpt is not None:
            latest_checkpoint = self.pretrained_ckpt

        if (self.exp_dir / "model_last.pt").exists():
            latest_checkpoint = self.exp_dir / "model_last.pt"
        else:
            ckpts = list(self.exp_dir.glob("*.pt"))
            if len(ckpts) > 0:
                latest_checkpoint = max(ckpts, key=lambda p: int(p.stem.removeprefix("model_")), default=None)

        if latest_checkpoint is None:
            logger.info("No checkpoint found, starting from scratch.")
            return 0

        checkpoint = torch.load(latest_checkpoint, weights_only=True, map_location="cpu")

        if self.is_main:
            self.ema_model.load_state_dict(checkpoint["ema_model"])

        if latest_checkpoint == self.pretrained_ckpt:
            checkpoint["model"] = {
                k.replace("ema_model.", ""): v
                for k, v in checkpoint["ema_model"].items()
                # FIXME: check if this is necessary
                # if k not in ["initted", "update", "step"]
            }
            self.model.load_state_dict(checkpoint["model"])
            update = 0
        else:
            self.model.load_state_dict(checkpoint["model"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.scheduler.load_state_dict(checkpoint["scheduler"])
            update = checkpoint["update"]

        return update

    def train(self, train_dataloader: DataLoader, cv_dataloader: DataLoader, seed: int = 42):
        generator = torch.Generator()
        generator.manual_seed(seed)

        self.optimizer = AdamW([p for p in self.model.parameters() if p.requires_grad], lr=self.learning_rate)
        total_updates = math.ceil(len(train_dataloader) / self.grad_accumulation_steps) * self.epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.num_warmup_updates,
            num_training_steps=total_updates,
        )
        current_update = self.load_checkpoint()

        self.accelerator.even_batches = False
        train_dataloader, cv_dataloader = self.accelerator.prepare(train_dataloader, cv_dataloader)
        self.model, self.optimizer, self.scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.scheduler
        )

        orig_epoch_step = len(train_dataloader)
        current_step = current_update * self.grad_accumulation_steps
        skipped_epoch = int(current_step // orig_epoch_step)

        for epoch in range(skipped_epoch, self.epochs):
            self.model.train()
            if epoch == skipped_epoch:
                skipped_batch = current_step % orig_epoch_step
                progress_bar_initial = math.ceil(skipped_batch / self.grad_accumulation_steps)
                current_dataloader = self.accelerator.skip_first_batches(train_dataloader, num_batches=skipped_batch)
            else:
                progress_bar_initial = 0
                current_dataloader = train_dataloader

            # Set epoch for the batch sampler if it exists
            if hasattr(train_dataloader, "batch_sampler") and hasattr(train_dataloader.batch_sampler, "set_epoch"):
                train_dataloader.batch_sampler.set_epoch(epoch)

            progress_bar = tqdm(
                range(math.ceil(len(train_dataloader) / self.grad_accumulation_steps)),
                desc=f"Epoch {epoch + 1}/{self.epochs}",
                unit="update",
                disable=not self.accelerator.is_local_main_process,
                initial=progress_bar_initial,
            )

            for batch in current_dataloader:
                with self.accelerator.accumulate(self.model):
                    loss, other = self.model(**batch)
                    info = other.get("info", {})
                    self.accelerator.backward(loss)

                    if self.grad_norm > 0 and self.accelerator.sync_gradients:
                        grad_norm = self.accelerator.clip_grad_norm_(self.model.parameters(), self.grad_norm)
                        if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                            logger.warning("Gradient norm is NaN of INF. Skipping update.")
                            self.optimizer.zero_grad()
                            continue

                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                if self.accelerator.sync_gradients:
                    if self.is_main:
                        self.ema_model.update()

                    current_update += 1
                    progress_bar.update(1)
                    progress_bar.set_postfix(update=str(current_update), loss=loss.item(), **info)

                if self.accelerator.is_local_main_process:
                    self.accelerator.log(
                        {"loss": loss.item(), "lr": self.scheduler.get_last_lr()[0], **info},
                        step=current_update,
                    )

                if current_update % self.save_per_updates == 0 and self.accelerator.sync_gradients:
                    cv_info = self.cv(cv_dataloader)
                    self.accelerator.log({f"cv_{key}": value for key, value in cv_info.items()}, step=current_update)
                    progress_bar.update()
                    progress_bar.set_postfix(update=str(current_update), **cv_info)
                    self.save_checkpoint(current_update)
                    self.model.train()

        self.save_checkpoint(current_update, is_last=True)
        self.accelerator.end_training()

    @torch.inference_mode()
    def cv(self, cv_dataloader: DataLoader):
        self.model.eval()
        metric2preds = {}
        metric2refs = {}
        for batch in tqdm(cv_dataloader, desc="Validation", disable=not self.accelerator.is_local_main_process):
            loss, other = self.model(**batch)
            batch_metric2preds = other.get("metric2pred", {})
            for name in batch_metric2preds.keys():
                if name not in metric2preds:
                    metric2preds[name] = []
                    metric2refs[name] = []
                preds = batch_metric2preds[name].detach().cpu().tolist()
                refs = batch["metrics"][name].detach().cpu().tolist()
                for pred, ref, sample_id, system_id in zip(preds, refs, batch["sample_ids"], batch["system_ids"]):
                    metric2preds[name].append({"sample_id": sample_id, "system_id": system_id, "value": pred})
                    metric2refs[name].append({"sample_id": sample_id, "system_id": system_id, "value": ref})

        self.accelerator.wait_for_everyone()
        info = {}
        for name in metric2preds.keys():
            if not self.accelerator.is_main_process:
                dist.gather_object(metric2preds[name], dst=0)
                dist.gather_object(metric2refs[name], dst=0)
                continue

            gathered = [None] * self.accelerator.num_processes
            dist.gather_object(metric2preds[name], gathered, dst=self.accelerator.process_index)
            metric2preds[name] = [item for lst in gathered for item in lst]
            dist.gather_object(metric2refs[name], gathered, dst=self.accelerator.process_index)
            metric2refs[name] = [item for lst in gathered for item in lst]
            info = other.get("info", {})
            info["loss"] = loss.detach().item()
            for name in metric2preds.keys():
                corr = calculate_metrics(metric2preds[name], metric2refs[name])
                for mode in ["utt", "sys"]:
                    for key, value in corr[mode].items():
                        info[f"{mode}_{key}_{name}"] = value
        return info
