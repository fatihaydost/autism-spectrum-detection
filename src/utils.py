from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict

import torch

from . import config


def get_device(force_cpu: bool = False) -> torch.device:
    if force_cpu:
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    stats: Dict[str, Any],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "stats": stats,
        "class_names": config.CLASS_NAMES,
    }
    torch.save(payload, path)


def load_checkpoint(path: Path, map_location: torch.device | None = None) -> Dict[str, Any]:
    checkpoint = torch.load(path, map_location=map_location)
    return checkpoint


@dataclass
class TrainStats:
    epoch: int
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def append_history(stats: TrainStats, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            history = json.load(handle)
    history.append(stats.to_dict())
    with path.open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
