import timm
import torch
from torch import nn


class GazeRegressor(nn.Module):
    """Pretrained backbone with a linear head regressing gaze angles."""

    def __init__(self, backbone: str = "resnet18", pretrained: bool = True):
        """Build the model.

        Args:
            backbone: Any timm model name, e.g. ``Config.backbone``.
            pretrained: Load ImageNet weights. Turn off in tests to avoid the
                download.
        """
        super().__init__()
        self.net = timm.create_model(backbone, pretrained=pretrained, num_classes=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``(B, 3, H, W)`` images to ``(B, 2)`` ``(yaw, pitch)`` in radians."""
        return self.net(x)
