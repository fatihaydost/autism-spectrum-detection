import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config, data, modeling, training, utils  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train autism classifier")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=config.NUM_WORKERS)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--cpu", action="store_true", help="Force training on CPU")
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=config.DEFAULT_CHECKPOINT_PATH,
    )
    parser.add_argument(
        "--history-path",
        type=Path,
        default=config.ARTIFACTS_DIR / "training_history.json",
    )
    parser.add_argument("--seed", type=int, default=config.SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_root = config.DATA_ROOT
    required_splits = ["Train", "valid"]
    missing_splits = [
        split for split in required_splits if not (data_root / split).exists()
    ]
    if missing_splits:
        missing = ", ".join(missing_splits)
        message = (
            "Veri dizini bulunamadı veya eksik: "
            f"{data_root}. Eksik klasörler: {missing}. "
            "Kaggle verisini indirip README'deki adımlarla "
            "`data/processed/asd_faces/` altında Train/valid[/Test] klasörlerini oluşturun."
        )
        raise SystemExit(message)

    data.set_seed(args.seed)
    # Veri yükleyicilerindeki rastgelelikler için tek bir noktadan tohum veriyoruz.
    loaders = data.create_dataloaders(
        batch_size=args.batch_size, num_workers=args.num_workers
    )

    train_loader = loaders.train
    val_loader = loaders.val
    test_loader = loaders.test

    model = modeling.create_resnet18()
    if args.freeze_backbone:
        modeling.freeze_backbone(model)

    device = utils.get_device(force_cpu=args.cpu)
    model = model.to(device)

    class_weights = data.compute_class_weights(train_loader.dataset)
    class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, verbose=True
    )

    best_stats = training.fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        scheduler=scheduler,
        epochs=args.epochs,
        checkpoint_path=args.checkpoint_path,
        history_path=args.history_path,
    )

    print(f"\nBest validation metrics: {best_stats}")

    if test_loader is not None and args.checkpoint_path.exists():
        print("\nEvaluating best model on Test split...")
        checkpoint = utils.load_checkpoint(args.checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        test_result = training.evaluate(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=device,
            desc="Test",
        )
        print(
            f"Test loss {test_result.loss:.4f} | "
            f"Test accuracy {test_result.acc:.3%}"
        )


if __name__ == "__main__":
    main()
