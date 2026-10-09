from dataclasses import dataclass
from pathlib import Path

from omegaconf import OmegaConf


@dataclass
class Config:
    seed: int = 42
    data_root: str = "data/MPIIFaceGaze/Data"
    backbone: str = "resnet18"
    img_size: int = 224
    batch_size: int = 64
    workers: int = 4
    epochs: int = 30
    lr: float = 1e-4
    weight_decay: float = 1e-2
    output_dir: str = "runs"


def load_config(path: str | Path, overrides: list[str] | None = None) -> Config:
    merged = OmegaConf.merge(
        OmegaConf.structured(Config),
        OmegaConf.load(path),
        OmegaConf.from_dotlist(overrides or []),
    )
    cfg = OmegaConf.to_object(merged)
    assert isinstance(cfg, Config)
    return cfg