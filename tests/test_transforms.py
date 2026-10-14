import pytest
import torch
from PIL import Image

from gazelle.data.transforms import (
    NATIVE_SIZE,
    build_eval_transform,
    build_train_transform,
)


@pytest.fixture
def white_image():
    return Image.new("RGB", (NATIVE_SIZE, NATIVE_SIZE), (255, 255, 255))


def test_output_is_chw_float32(white_image):
    out = build_eval_transform("resnet18")(white_image)
    assert out.shape == (3, NATIVE_SIZE, NATIVE_SIZE)
    assert out.dtype == torch.float32


def test_imagenet_backbone_uses_imagenet_stats(white_image):
    """White maps to (1 - mean) / std."""
    out = build_eval_transform("resnet18")(white_image)
    assert out[0].mean().item() == pytest.approx((1 - 0.485) / 0.229, abs=1e-3)


def test_vit_uses_its_own_stats(white_image):
    """vit_base_patch16_224 is normalised with 0.5/0.5, not ImageNet."""
    out = build_eval_transform("vit_base_patch16_224")(white_image)
    assert out[0].mean().item() == pytest.approx(1.0, abs=1e-3)


def test_backbone_input_size_is_respected(white_image):
    """efficientnet_b3 wants 288, so the pipeline must resize."""
    out = build_eval_transform("efficientnet_b3")(white_image)
    assert out.shape == (3, 288, 288)


def test_unknown_backbone_rejected():
    with pytest.raises(ValueError, match="Unknown timm backbone"):
        build_eval_transform("not_a_real_backbone")


def test_eval_transform_is_deterministic(white_image):
    tf = build_eval_transform("resnet18")
    assert torch.equal(tf(white_image), tf(white_image))


def test_train_transform_has_no_geometric_augmentation(white_image):
    """Flips, rotations and crops would invalidate the gaze label."""
    tf = build_train_transform("resnet18")
    assert torch.equal(tf(white_image), tf(white_image))
