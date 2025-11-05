from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from torch.cuda.amp import GradScaler, autocast

from . import config, utils


@dataclass
class EpochResult:
    loss: float
    acc: float


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
    scaler: GradScaler | None = None,
) -> EpochResult:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    use_amp = scaler is not None and scaler.is_enabled()

    progress = tqdm(loader, desc="Train", leave=False)
    for images, targets, _ in progress:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, targets)
        loss_value = float(loss.detach())

        # AMP aktifken backward+step işlemlerini ölçekleyerek taşma riskini azaltıyoruz.
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss_value * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

        progress.set_postfix(loss=f"{loss_value:.4f}")

    epoch_loss = running_loss / max(total, 1)
    epoch_acc = correct / max(total, 1)
    return EpochResult(epoch_loss, epoch_acc)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    desc: str = "Val",
    use_amp: bool = False,
) -> EpochResult:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    progress = tqdm(loader, desc=desc, leave=False)
    for images, targets, _ in progress:
        images = images.to(device)
        targets = targets.to(device)

        with autocast(enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, targets)
        loss_value = float(loss.detach())

        # Validasyon döngüsünde ağırlık güncellemesi yapılmadığından sadece istatistik topluyoruz.
        running_loss += loss_value * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

        progress.set_postfix(loss=f"{loss_value:.4f}")

    epoch_loss = running_loss / max(total, 1)
    epoch_acc = correct / max(total, 1)
    return EpochResult(epoch_loss, epoch_acc)


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
    scheduler: Optional[_LRScheduler] = None,
    epochs: int = 10,
    checkpoint_path: Path | None = None,
    history_path: Path | None = None,
) -> Dict[str, float]:
    best_val_acc = 0.0
    best_stats: Dict[str, float] = {}
    scaler = GradScaler(enabled=device.type == "cuda")

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        train_result = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            scaler=scaler,
        )
        val_result = evaluate(
            model,
            val_loader,
            criterion,
            device,
            use_amp=scaler.is_enabled(),
        )

        if scheduler is not None:
            scheduler.step(val_result.loss)

        stats = utils.TrainStats(
            epoch=epoch,
            train_loss=train_result.loss,
            train_acc=train_result.acc,
            val_loss=val_result.loss,
            val_acc=val_result.acc,
        )

        print(
            f"Train loss {train_result.loss:.4f} | "
            f"Train acc {train_result.acc:.3%} | "
            f"Val loss {val_result.loss:.4f} | "
            f"Val acc {val_result.acc:.3%}"
        )

        if history_path:
            utils.append_history(stats, history_path)

        if val_result.acc >= best_val_acc:
            best_val_acc = val_result.acc
            best_stats = stats.to_dict()
            if checkpoint_path:
                utils.save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    stats=best_stats,
                    path=checkpoint_path,
                )
                print(f"Saved checkpoint to {checkpoint_path}")

    return best_stats
