from dataclasses import asdict
from itertools import pairwise

import lightning as L
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from gazelle.config import Config
from gazelle.module import GazeModule

CFG = Config(pretrained=False, lr=0.05, weight_decay=0.123)


def tiny_module(cfg: Config = CFG) -> GazeModule:
    """GazeModule with the backbone swapped out, so tests stay fast."""
    module = GazeModule(asdict(cfg))
    module.model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 2))
    return module


def one_batch_loader(n: int = 8) -> DataLoader:
    torch.manual_seed(0)
    images = torch.randn(n, 3, 8, 8)
    targets = torch.rand(n, 2) * 0.4 - 0.2
    return DataLoader(TensorDataset(images, targets), batch_size=n)


def silent_trainer(**kwargs) -> L.Trainer:
    return L.Trainer(
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        **kwargs,
    )


def test_forward_returns_two_angles():
    assert tiny_module()(torch.randn(4, 3, 8, 8)).shape == (4, 2)


def test_overfits_a_single_batch():
    """The highest-value wiring test.

    If the model cannot drive loss to ~0 on eight samples, something is broken
    — the optimizer is not stepping, or inputs and targets are misaligned.
    """
    L.seed_everything(0, workers=True)
    module = tiny_module()
    # MSE, not the L1 default: Adam steps are ~lr regardless of gradient size,
    # so L1's constant gradient oscillates around the optimum at amplitude ~lr.
    module.loss_fn = nn.MSELoss()

    loader = one_batch_loader()
    silent_trainer(max_epochs=200).fit(module, loader)

    with torch.no_grad():
        images, targets = next(iter(loader))
        assert module.loss_fn(module(images), targets).item() < 1e-3


def test_configure_optimizers_uses_config():
    config = tiny_module().configure_optimizers()
    optimizer = config["optimizer"]
    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.param_groups[0]["lr"] == CFG.lr
    assert optimizer.param_groups[0]["weight_decay"] == CFG.weight_decay


def test_lr_decays_to_zero_over_the_run():
    """Cosine decay must span exactly cfg.epochs."""
    config = tiny_module().configure_optimizers()
    optimizer, scheduler = config["optimizer"], config["lr_scheduler"]

    rates = [optimizer.param_groups[0]["lr"]]
    for _ in range(CFG.epochs):
        scheduler.step()
        rates.append(optimizer.param_groups[0]["lr"])

    assert rates[0] == pytest.approx(CFG.lr)
    assert rates[-1] == pytest.approx(0.0, abs=1e-9)
    assert all(a >= b for a, b in pairwise(rates))


@pytest.mark.parametrize("stage", ["train", "val", "test"])
def test_each_stage_logs_loss_and_error(stage):
    module = tiny_module()
    trainer = silent_trainer(max_epochs=1)
    loader = one_batch_loader()

    if stage == "train":
        trainer.fit(module, loader)
    elif stage == "val":
        trainer.validate(module, loader)
    else:
        trainer.test(module, loader)

    assert f"{stage}/loss" in trainer.logged_metrics
    assert f"{stage}/angular_error" in trainer.logged_metrics


def test_angular_error_is_in_degrees():
    module = tiny_module()
    trainer = silent_trainer()
    trainer.test(module, one_batch_loader())
    error = trainer.logged_metrics["test/angular_error"].item()
    assert 0 < error < 180


def test_checkpoint_round_trip(tmp_path):
    """load_from_checkpoint must rebuild the model without being told the config.

    Uses the real backbone, since that is what gets reconstructed from the
    stored hyperparameters.
    """
    module = GazeModule(asdict(CFG))
    loader = DataLoader(
        TensorDataset(torch.randn(2, 3, 224, 224), torch.zeros(2, 2)), batch_size=2
    )
    trainer = silent_trainer(max_epochs=1)
    trainer.fit(module, loader)

    path = tmp_path / "ckpt.ckpt"
    trainer.save_checkpoint(path)

    restored = GazeModule.load_from_checkpoint(path, map_location="cpu")
    assert restored.cfg == CFG
    for a, b in zip(module.model.parameters(), restored.model.parameters()):
        assert torch.equal(a, b)


def test_seed_makes_runs_identical():
    def run_once() -> float:
        L.seed_everything(123, workers=True)
        module = tiny_module()
        trainer = silent_trainer(max_epochs=1)
        trainer.fit(module, one_batch_loader())
        return trainer.logged_metrics["train/loss"].item()

    assert run_once() == run_once()
