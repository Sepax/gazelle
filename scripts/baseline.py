"""Constant-predictor baseline for leave-one-person-out gaze estimation.

Scores a predictor that ignores the image entirely and always outputs the mean
gaze direction of the training participants. This bounds how much of the task
is solvable from the fact that laptop users mostly look slightly downward, and
so bounds how much credit a trained model can claim for actually using the
face.

Only label files are read, so this needs no GPU and runs in seconds.

    uv run python scripts/baseline.py
    uv run python scripts/baseline.py --folds 0 1 2 --out baseline.json
"""

import argparse
import json
import statistics
from pathlib import Path

import torch

from gazelle.data.dataset import GazeDataset
from gazelle.data.splits import lopo_split
from gazelle.metrics import angular_error

DEFAULT_ROOT = "data/processed/MPIIFaceGaze"
NUM_SUBJECTS = 15


def read_labels(data_root: str, subjects: list[int]) -> torch.Tensor:
    """Return the (yaw, pitch) labels of the given participants, in radians.

    GazeDataset indexes the label files in its constructor and only decodes an
    image when a sample is actually requested, so reading `.samples` touches no
    image data. The transform is never applied for the same reason; identity is
    passed only because the constructor requires one.
    """
    dataset = GazeDataset(data_root, subjects, transform=lambda image: image)
    return torch.tensor([gaze for _, gaze in dataset.samples])


def constant_predictor_error(data_root: str, test_subject: int) -> float:
    """Mean angular error, in degrees, of predicting the training mean gaze.

    The constant comes from the 14 training participants only. Labels of the
    held-out participant are used solely to score the prediction, exactly as
    they are when scoring a trained model.
    """
    split = lopo_split(test_subject, num_subjects=NUM_SUBJECTS)

    constant = read_labels(data_root, split.train).mean(dim=0, keepdim=True)
    test_labels = read_labels(data_root, split.test)

    # Same metric the model is scored with, so the two numbers are comparable.
    errors = angular_error(constant.expand_as(test_labels), test_labels)
    return float(errors.mean())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", default=DEFAULT_ROOT)
    parser.add_argument(
        "--folds", type=int, nargs="*", default=None, help="Defaults to all 15"
    )
    parser.add_argument("--out", type=Path, default=None, help="Optional JSON output")
    args = parser.parse_args()

    folds = list(range(NUM_SUBJECTS)) if args.folds is None else args.folds
    per_fold = {
        f"p{subject:02d}": constant_predictor_error(args.data_root, subject)
        for subject in folds
    }
    errors = list(per_fold.values())

    summary = {
        "mean": statistics.fmean(errors),
        "std": statistics.stdev(errors) if len(errors) > 1 else 0.0,
        "folds": per_fold,
    }

    print("=== constant-predictor baseline ===")
    for name, error in per_fold.items():
        print(f"{name}: {error:6.2f} deg")
    print(f"mean: {summary['mean']:.2f} +/- {summary['std']:.2f} deg")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"written: {args.out}")


if __name__ == "__main__":
    main()
