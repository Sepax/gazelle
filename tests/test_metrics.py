import math

import numpy as np
import pytest
import torch

from gazelle.data.gazehub import data_processing_core as dpc
from gazelle.metrics import angular_error, gaze_to_3d

STRAIGHT_AHEAD = torch.tensor([[0.0, 0.0]])


def test_gaze_to_3d_shape():
    assert gaze_to_3d(torch.zeros(5, 2)).shape == (5, 3)


def test_gaze_to_3d_returns_unit_vectors():
    angles = torch.tensor([[0.0, 0.0], [0.3, -0.2], [-0.5, 0.4]])
    norms = gaze_to_3d(angles).norm(dim=1)
    assert torch.allclose(norms, torch.ones(3), atol=1e-5)


def test_gaze_to_3d_straight_ahead():
    """Zero angles look down the negative z axis, per the GazeHub convention."""
    assert torch.allclose(
        gaze_to_3d(STRAIGHT_AHEAD), torch.tensor([[0.0, 0.0, -1.0]]), atol=1e-6
    )


@pytest.mark.parametrize(
    "vec",
    [
        (0.0, 0.0, -1.0),
        (0.25, 0.05, -0.97),
        (-0.30, -0.12, -0.95),
        (0.10, -0.40, -0.91),
    ],
)
def test_round_trips_through_gazehub_conversion(vec):
    """``gaze_to_3d`` must invert ``GazeTo2d``, which produced the labels.

    Pins the sign convention against the benchmark code itself, so a flipped
    axis cannot pass.
    """
    v = np.array(vec, dtype=np.float64)
    v /= np.linalg.norm(v)

    angles = torch.from_numpy(dpc.GazeTo2d(v)).to(torch.float32).unsqueeze(0)
    expected = torch.from_numpy(v).to(torch.float32).unsqueeze(0)

    assert torch.allclose(gaze_to_3d(angles), expected, atol=1e-5)


def test_identical_directions_give_zero_error():
    angles = torch.tensor([[0.2, -0.1], [0.0, 0.0], [-0.4, 0.3]])
    err = angular_error(angles, angles.clone())
    assert torch.allclose(err, torch.zeros(3), atol=1e-4)


def test_identical_directions_do_not_produce_nan():
    """The classic bug: arccos of a dot product nudged past 1.0 by rounding."""
    angles = torch.tensor([[0.3, 0.3]]).repeat(64, 1)
    assert not torch.isnan(angular_error(angles, angles.clone())).any()


def test_result_is_in_degrees():
    """Yaw of pi/2 is 90 degrees away from straight ahead, not 1.57."""
    quarter_turn = torch.tensor([[math.pi / 2, 0.0]])
    assert angular_error(STRAIGHT_AHEAD, quarter_turn).item() == pytest.approx(90.0, abs=1e-3)


def test_pitch_is_measured_too():
    quarter_turn = torch.tensor([[0.0, math.pi / 2]])
    assert angular_error(STRAIGHT_AHEAD, quarter_turn).item() == pytest.approx(90.0, abs=1e-3)


def test_known_forty_five_degrees():
    angles = torch.tensor([[math.pi / 4, 0.0]])
    assert angular_error(STRAIGHT_AHEAD, angles).item() == pytest.approx(45.0, abs=1e-3)


def test_is_symmetric():
    a = torch.tensor([[0.2, -0.1], [0.5, 0.3]])
    b = torch.tensor([[-0.3, 0.4], [0.1, 0.1]])
    assert torch.allclose(angular_error(a, b), angular_error(b, a), atol=1e-5)


def test_returns_one_error_per_sample():
    a = torch.zeros(7, 2)
    b = torch.rand(7, 2) * 0.2
    assert angular_error(a, b).shape == (7,)


def test_error_is_non_negative():
    a = torch.rand(32, 2) - 0.5
    b = torch.rand(32, 2) - 0.5
    assert (angular_error(a, b) >= 0).all()
