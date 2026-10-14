from dataclasses import dataclass
from pathlib import Path

from omegaconf import OmegaConf


@dataclass
class Config:
    seed: int = 42
    data_root: str = "data/processed/MPIIFaceGaze"
    test_subject: int = 0
    backbone: str = "resnet18"
    pretrained: bool = True
    val_fraction: float = 0.05
    batch_size: int = 64
    workers: int = 4
    epochs: int = 30
    lr: float = 1e-4
    weight_decay: float = 1e-2
    output_dir: str = "runs"
    benchmark_dir: str = "benchmarks"


def load_config(path: str | Path, overrides: list[str] | None = None) -> Config:
    merged = OmegaConf.merge(
        OmegaConf.structured(Config),
        OmegaConf.load(path),
        OmegaConf.from_dotlist(overrides or []),
    )
    cfg = OmegaConf.to_object(merged)
    assert isinstance(cfg, Config)
    return cfg


def save_config(cfg: Config, path: str | Path) -> None:
    """Write ``cfg`` beside a run's outputs as YAML.

    Records exactly what produced a run, including defaults never written in
    the source config and any CLI overrides. Reloadable with
    :func:`load_config`.

    Args:
        cfg: The resolved configuration.
        path: Destination ``.yaml``; parent directories are created.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.structured(cfg), path)
