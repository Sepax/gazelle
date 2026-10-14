import pytest
import torch
from conftest import gaze_for, make_data_root

from gazelle.data.dataset import GazeDataset
from gazelle.data.transforms import NATIVE_SIZE, build_eval_transform

BACKBONE = "resnet18"


@pytest.fixture
def transform():
    return build_eval_transform(BACKBONE)


def test_length_is_samples_times_participants(data_root, transform):
    assert len(GazeDataset(data_root, [0, 1, 2], transform)) == 12


def test_only_requested_participants_are_indexed(data_root, transform):
    ds = GazeDataset(data_root, [1], transform)
    assert len(ds) == 4
    assert all(path.startswith("p01/") for path, _ in ds.samples)


def test_header_row_is_skipped(data_root, transform):
    ds = GazeDataset(data_root, [0], transform)
    assert all(path.endswith(".jpg") for path, _ in ds.samples)


def test_index_is_flat_across_participants(data_root, transform):
    ds = GazeDataset(data_root, [0, 1, 2], transform)
    assert ds.samples[0][0] == "p00/face/1.jpg"
    assert ds.samples[4][0] == "p01/face/1.jpg"
    assert ds.samples[8][0] == "p02/face/1.jpg"


def test_label_is_yaw_then_pitch(data_root, transform):
    """2DGaze is (yaw, pitch); a swap here corrupts every downstream metric."""
    ds = GazeDataset(data_root, [1], transform)
    _, gaze = ds[2]
    expected_yaw, expected_pitch = gaze_for(1, 3)
    assert gaze[0].item() == pytest.approx(expected_yaw)
    assert gaze[1].item() == pytest.approx(expected_pitch)
    assert gaze[0] > 0 > gaze[1]


def test_getitem_shapes_and_dtypes(data_root, transform):
    image, gaze = GazeDataset(data_root, [0], transform)[0]
    assert image.shape == (3, NATIVE_SIZE, NATIVE_SIZE)
    assert image.dtype == torch.float32
    assert gaze.shape == (2,)
    assert gaze.dtype == torch.float32


def test_transform_is_applied(data_root):
    ds = GazeDataset(data_root, [0], transform=lambda img: torch.zeros(1))
    image, _ = ds[0]
    assert image.shape == (1,)


def test_invalid_root_rejected(tmp_path, transform):
    with pytest.raises(ValueError):
        GazeDataset(tmp_path / "missing", [0], transform)


def test_missing_label_file_rejected(data_root, transform):
    with pytest.raises(FileNotFoundError):
        GazeDataset(data_root, [9], transform)


def test_no_images_needed_until_getitem(tmp_path, transform):
    """Indexing reads label files only; images load lazily."""
    root = make_data_root(tmp_path, subjects=[0], write_images=False)
    ds = GazeDataset(root, [0], transform)
    assert len(ds) == 4
    with pytest.raises(FileNotFoundError):
        ds[0]
