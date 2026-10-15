import argparse
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, Subset, random_split

from gazelle.config import Config, load_config, save_config
from gazelle.data.dataset import GazeDataset
from gazelle.data.splits import lopo_split
from gazelle.data.transforms import build_eval_transform, build_train_transform
from gazelle.logging_setup import quiet_startup
from gazelle.module import GazeModule

torch.set_float32_matmul_precision("high")


def build_loaders(cfg: Config) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build the train, val and test loaders for one fold.

    Two splits happen here. ``lopo_split`` holds out one participant, then the
    remaining 14 are split by *frame* into train and val.

    Args:
        cfg: Supplies ``data_root``, ``test_subject``, ``backbone``,
            ``val_fraction``, ``batch_size`` and ``workers``.

    Returns:
        ``(train_loader, val_loader, test_loader)``. Only the train loader
        shuffles.
    """
    split = lopo_split(cfg.test_subject)
    eval_tf = build_eval_transform(cfg.backbone)

    # Same 14 participants twice, so the val half is read without augmentation.
    augmented = GazeDataset(cfg.data_root, split.train, build_train_transform(cfg.backbone))
    plain = GazeDataset(cfg.data_root, split.train, eval_tf)

    generator = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = random_split(
        augmented, [1 - cfg.val_fraction, cfg.val_fraction], generator=generator
    )

    kwargs = {
        "batch_size": cfg.batch_size,
        "num_workers": cfg.workers,
        # Without this, every loader re-forks its workers each epoch.
        "persistent_workers": cfg.workers > 0,
    }
    return (
        DataLoader(train_set, shuffle=True, **kwargs),
        DataLoader(Subset(plain, val_set.indices), **kwargs),
        DataLoader(GazeDataset(cfg.data_root, split.test, eval_tf), **kwargs),
    )


def run_dir_for(cfg: Config, root: str | None = None) -> Path:
    """Timestamped output directory for one fold, so reruns do not collide."""
    stamp = datetime.now(UTC).astimezone().strftime("%m%d-%H%M")
    base = Path(root if root is not None else cfg.output_dir)
    return base / f"p{cfg.test_subject:02d}_{cfg.backbone}_{stamp}"


def run(cfg: Config) -> dict[str, float]:
    """Train one fold and return its test metrics.

    Do not early-stop on validation: the val frames come from participants the
    model trains on, so val error reads optimistically low. Train the fixed
    number of epochs.

    Args:
        cfg: The run configuration.

    Returns:
        ``{"test/loss": ..., "test/angular_error": ...}`` for the held-out
        participant, scored with the final-epoch weights.
    """
    L.seed_everything(cfg.seed, workers=True)

    run_dir = run_dir_for(cfg)
    save_config(cfg, run_dir / "config.yaml")

    train_loader, val_loader, test_loader = build_loaders(cfg)
    module = GazeModule(asdict(cfg))

    checkpoint = ModelCheckpoint(
        dirpath=run_dir / "checkpoints",
        filename="best",
        monitor="val/angular_error",
        mode="min",
        save_last=True,
    )
    trainer = L.Trainer(
        max_epochs=cfg.epochs,
        default_root_dir=run_dir,
        logger=TensorBoardLogger(save_dir=run_dir, name="", version=""),
        callbacks=[checkpoint, LearningRateMonitor(logging_interval="epoch")],
        accelerator="auto",
        log_every_n_steps=25,
    )

    trainer.fit(module, train_loader, val_loader)

    # Score the final epoch, not the best-validation one. Val frames come from
    # participants the model trains on, so picking the epoch that minimises val
    # error selects on a signal that does not track cross-person performance.
    # best.ckpt is still saved, as a diagnostic.
    results = trainer.test(module, test_loader, ckpt_path=None)
    return dict(results[0])


def main() -> None:
    """CLI entry point.

    Parses ``--config`` plus trailing dotlist overrides (``epochs=1``), loads
    the config and calls :func:`run`.
    """
    parser = argparse.ArgumentParser(description="Train one leave-one-person-out fold.")
    parser.add_argument("--config", default="configs/resnet18.yaml")
    parser.add_argument(
        "overrides", nargs="*", help="Dotlist overrides, e.g. epochs=1 batch_size=8"
    )
    args = parser.parse_args()

    quiet_startup()
    run(load_config(args.config, args.overrides))


if __name__ == "__main__":
    main()
