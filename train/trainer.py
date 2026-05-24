from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.config import Config


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        cfg: Config,
        device: torch.device,
        checkpoint_path: Path | None = None,
        epoch_log_path: Path | None = None,
    ) -> None:
        self.model = model.to(device)
        self.cfg = cfg
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.checkpoint_path = checkpoint_path
        self.epoch_log_path = epoch_log_path
        self.best_val_acc = 0.0

        self.optimizer = None
        self.scheduler = None
        self.scaler = torch.amp.GradScaler("cuda", enabled=cfg.use_amp)

    def setup_optim(self, optimizer, scheduler) -> None:
        self.optimizer = optimizer
        self.scheduler = scheduler

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        for images, targets in loader:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=self.cfg.use_amp):
                logits = self.model(images)
                loss = self.criterion(logits, targets)

            total_loss += loss.item() * targets.size(0)
            correct += (logits.argmax(1) == targets).sum().item()
            total += targets.size(0)

        return total_loss / total, correct / total

    def train_one_epoch(self, loader: DataLoader, epoch: int) -> tuple[float, float]:
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch}/{self.cfg.epochs}", leave=False)
        for images, targets in pbar:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=self.cfg.use_amp):
                logits = self.model(images)
                loss = self.criterion(logits, targets)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item() * targets.size(0)
            correct += (logits.argmax(1) == targets).sum().item()
            total += targets.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        return total_loss / total, correct / total

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> dict[str, float]:
        history: dict[str, float] = {}

        for epoch in range(1, int(self.cfg.epochs) + 1):
            train_loss, train_acc = self.train_one_epoch(train_loader, epoch)
            val_loss, val_acc = self.evaluate(val_loader)

            if self.scheduler is not None:
                self.scheduler.step()

            lr = self.optimizer.param_groups[0]["lr"]
            line = (
                f"[{epoch:03d}/{self.cfg.epochs}] "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | lr={lr:.6f}"
            )
            print(line, flush=True)
            if self.epoch_log_path is not None:
                with self.epoch_log_path.open("a", encoding="utf-8") as f:
                    f.write(line + "\n")

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                if self.checkpoint_path is not None:
                    self._save_checkpoint(epoch, val_acc)

            history["last_train_acc"] = train_acc
            history["last_val_acc"] = val_acc

        history["best_val_acc"] = self.best_val_acc
        return history

    def _save_checkpoint(self, epoch: int, val_acc: float) -> None:
        assert self.checkpoint_path is not None
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "epoch": epoch,
                "val_acc": val_acc,
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "config": self.cfg.to_dict(),
            },
            self.checkpoint_path,
        )

    def load_checkpoint(self, path: Path) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state"])
        if self.optimizer is not None and "optimizer_state" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.best_val_acc = float(ckpt.get("val_acc", 0.0))
