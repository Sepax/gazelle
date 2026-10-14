from typing import Any

import lightning as L
import torch
from torch import nn

from gazelle import metrics
from gazelle.config import Config
from gazelle.models import build_model


class GazeModule(L.LightningModule):
    """Lightning wrapper around the gaze regressor.

    Holds the loss, the metric and the optimizer, so the training loop itself
    is Lightning's. Takes a plain dict rather than a :class:`Config` so that
    ``save_hyperparameters`` can round-trip it through a checkpoint.
    """

    def __init__(self, config: dict[str, Any]):
        """Build the model described by ``config``.

        Args:
            config: ``dataclasses.asdict(cfg)``. Stored as a hyperparameter,
                which is what lets ``load_from_checkpoint`` rebuild the right
                architecture without being told.
        """
        super().__init__()
        self.save_hyperparameters()
        self.cfg = Config(**config)
        self.model = build_model(self.cfg)
        self.loss_fn = nn.L1Loss()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Map ``(B, 3, H, W)`` images to ``(B, 2)`` ``(yaw, pitch)`` in radians."""
        return self.model(images)

    def _step(self, batch: tuple[torch.Tensor, torch.Tensor], stage: str) -> torch.Tensor:
        """Shared logic: predict, score, log. Loss is radians, error is degrees."""
        images, targets = batch
        preds = self(images)

        loss = self.loss_fn(preds, targets)
        error = metrics.angular_error(preds, targets).mean()

        self.log(f"{stage}/loss", loss, prog_bar=True)
        self.log(f"{stage}/angular_error", error, prog_bar=stage != "train")
        return loss

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx: int) -> torch.Tensor:
        return self._step(batch, "val")

    def test_step(self, batch, batch_idx: int) -> torch.Tensor:
        return self._step(batch, "test")

    def configure_optimizers(self) -> dict[str, Any]:
        """AdamW with cosine decay over the full run.

        The schedule matters more than usual here: Adam takes steps of roughly
        the learning rate regardless of gradient magnitude, and L1 has a
        constant gradient, so at a fixed rate the loss oscillates around the
        optimum at amplitude ~lr instead of settling. Decaying to zero lets the
        last epochs actually converge.
        """
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.cfg.epochs
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
