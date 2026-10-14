from collections.abc import Callable

import timm
import torch
from PIL import Image
from torchvision.transforms import v2

# Side length the GazeHub preprocessing emits; resizing is skipped at this size.
NATIVE_SIZE = 224

TransformFn = Callable[[Image.Image], torch.Tensor]


def _base_steps(backbone: str) -> list[v2.Transform]:
    """Build the steps turning a PIL crop into the tensor ``backbone`` expects.

    Size, mean, std and interpolation come from timm's pretrained config, so
    switching backbone adapts the pipeline automatically.

    Raises:
        ValueError: If timm has no pretrained config for ``backbone``.
    """
    cfg = timm.get_pretrained_cfg(backbone)
    if cfg is None:
        raise ValueError(f"Unknown timm backbone: {backbone}")

    size = cfg.input_size[-1]
    steps: list[v2.Transform] = []
    if size != NATIVE_SIZE:
        mode = v2.InterpolationMode(cfg.interpolation)
        steps.append(v2.Resize((size, size), interpolation=mode, antialias=True))
    steps += [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=list(cfg.mean), std=list(cfg.std)),
    ]
    return steps


def build_eval_transform(backbone: str) -> TransformFn:
    """Deterministic pipeline for validation and test."""
    return v2.Compose(_base_steps(backbone))


def build_train_transform(backbone: str) -> TransformFn:
    """Training pipeline.

    Identical to the eval pipeline for now. Augmentation belongs here and
    nowhere else, and must be appearance-only: the label is a gaze direction,
    so flips, rotations and crops invalidate it.
    """
    return v2.Compose(_base_steps(backbone))
