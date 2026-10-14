import argparse
import dataclasses
import json
import statistics
from datetime import UTC, datetime
from pathlib import Path

from gazelle.config import Config, load_config
from gazelle.train import run


def run_benchmark(cfg: Config, folds: list[int] | None = None) -> dict[str, float]:
    """Run leave-one-person-out across every fold and aggregate.

    This is the reportable protocol: 14 train / 1 test, repeated for all 15
    participants, mean angular error. Freeze every hyperparameter before
    running it — tuning against these numbers overfits the benchmark and makes
    them meaningless.

    Args:
        cfg: Frozen configuration; ``test_subject`` is overridden per fold.
        folds: Participants to hold out. Defaults to all 15.

    Returns:
        ``{"mean", "std", "folds", "config"}`` — errors in degrees, plus the
        configuration that produced them.
    """
    folds = list(range(15)) if folds is None else folds
    stamp = datetime.now(UTC).astimezone().strftime("%m%d-%H%M")
    bench_dir = Path(cfg.benchmark_dir) / stamp

    per_fold: dict[str, float] = {}
    for subject in folds:
        fold_cfg = dataclasses.replace(
            cfg, test_subject=subject, output_dir=str(bench_dir)
        )
        per_fold[f"p{subject:02d}"] = run(fold_cfg)["test/angular_error"]

    errors = list(per_fold.values())
    summary = {
        "mean": statistics.fmean(errors),
        "std": statistics.stdev(errors) if len(errors) > 1 else 0.0,
        "folds": per_fold,
        "config": dataclasses.asdict(cfg),
    }

    print("\n=== leave-one-person-out ===")
    for name, error in per_fold.items():
        print(f"{name}: {error:6.2f} deg")
    print(f"mean: {summary['mean']:.2f} +/- {summary['std']:.2f} deg")

    out = bench_dir / "summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"written: {out}")
    return summary


def main() -> None:
    """CLI entry point: run the benchmark and write its summary."""
    parser = argparse.ArgumentParser(description="Run the 15-fold LOPO benchmark.")
    parser.add_argument("--config", default="configs/resnet18.yaml")
    parser.add_argument(
        "--folds", type=int, nargs="*", default=None, help="Defaults to all 15"
    )
    parser.add_argument("overrides", nargs="*", help="Dotlist overrides")
    args = parser.parse_args()

    run_benchmark(load_config(args.config, args.overrides), args.folds)


if __name__ == "__main__":
    main()
