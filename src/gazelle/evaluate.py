import argparse

import lightning as L
from torch.utils.data import DataLoader

from gazelle.data.dataset import GazeDataset
from gazelle.data.transforms import build_eval_transform
from gazelle.logging_setup import quiet_startup
from gazelle.module import GazeModule


def evaluate_checkpoint(checkpoint: str, subject: int | None = None) -> dict[str, float]:
    """Score a saved checkpoint on one participant.

    The config stored in the checkpoint rebuilds the matching model and
    transform, so a checkpoint is scored exactly as it was trained.

    Args:
        checkpoint: Path to a ``.ckpt`` written by Lightning.
        subject: Participant to score. Defaults to the fold's held-out
            ``test_subject``.

    Returns:
        ``{"test/loss": ..., "test/angular_error": ...}``.
    """
    module = GazeModule.load_from_checkpoint(checkpoint)
    cfg = module.cfg

    dataset = GazeDataset(
        cfg.data_root,
        [cfg.test_subject if subject is None else subject],
        build_eval_transform(cfg.backbone),
    )
    loader = DataLoader(dataset, batch_size=cfg.batch_size, num_workers=cfg.workers)

    trainer = L.Trainer(accelerator="auto", logger=False, enable_checkpointing=False)
    return dict(trainer.test(module, loader)[0])


def main() -> None:
    """CLI entry point: score a checkpoint and print its angular error."""
    parser = argparse.ArgumentParser(description="Evaluate a saved checkpoint.")
    parser.add_argument("checkpoint", help="Path to a .ckpt file")
    parser.add_argument(
        "--subject", type=int, default=None, help="Defaults to the fold's test subject"
    )
    args = parser.parse_args()

    quiet_startup()
    metrics = evaluate_checkpoint(args.checkpoint, args.subject)
    print(f"angular_error: {metrics['test/angular_error']:.2f} deg")


if __name__ == "__main__":
    main()
