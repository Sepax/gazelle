from torch import nn

from gazelle.config import Config
from gazelle.models.regressor import GazeRegressor


def build_model(cfg: Config) -> nn.Module:
    """Construct the model described by ``cfg``.

    The single place that turns config into a model, so train.py, evaluate.py
    and benchmark.py cannot drift apart.

    Args:
        cfg: Holds ``backbone``; add a ``pretrained`` field if you want it
            configurable rather than always True.

    Returns:
        A model mapping ``(B, 3, H, W)`` to ``(B, 2)`` gaze angles.
    """
    # Device placement stays in train.py, so this is testable on CPU.
    return GazeRegressor(backbone=cfg.backbone, pretrained=cfg.pretrained)
