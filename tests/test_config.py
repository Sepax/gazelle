import pytest

from gazelle.config import Config, load_config


def test_yaml_values_are_loaded(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("backbone: convnext_tiny\nlr: 1.0e-3\n")
    cfg = load_config(path)
    assert cfg.backbone == "convnext_tiny"
    assert cfg.lr == 1e-3


def test_missing_keys_use_defaults(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("epochs: 3\n")
    cfg = load_config(path)
    assert cfg.epochs == 3
    assert cfg.batch_size == Config().batch_size


def test_overrides_beat_yaml(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("epochs: 3\n")
    cfg = load_config(path, ["epochs=7", "test_subject=5"])
    assert cfg.epochs == 7
    assert cfg.test_subject == 5


def test_unknown_key_rejected(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("not_a_field: 1\n")
    with pytest.raises(Exception):
        load_config(path)