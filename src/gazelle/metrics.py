import torch


def gaze_to_3d(angles: torch.Tensor) -> torch.Tensor:
    """Convert ``(yaw, pitch)`` angles to unit gaze vectors.

    Must invert ``GazeTo2d`` in ``data/gazehub/data_processing_core.py``, which
    the preprocessing used to write the ``2DGaze`` labels. Derive it from those
    two lines rather than guessing the sign convention.

    Args:
        angles: ``(B, 2)`` float tensor of ``(yaw, pitch)`` in radians.

    Returns:
        ``(B, 3)`` unit vectors.
    """
    yaw, pitch = angles.unbind(dim=-1)
    x = -torch.sin(yaw) * torch.cos(pitch)
    y = -torch.sin(pitch)
    z = -torch.cos(yaw) * torch.cos(pitch)
    return torch.stack((x, y, z), dim=-1)


def angular_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Angle between predicted and true gaze directions, in degrees.

    The headline metric: a bug here corrupts every reported result and nothing
    downstream will catch it. Clamp the dot product to ``[-1, 1]`` before
    ``arccos`` — floating-point error pushes it just past 1.0 and yields NaN.

    Args:
        pred: ``(B, 2)`` predicted ``(yaw, pitch)`` in radians.
        target: ``(B, 2)`` ground-truth ``(yaw, pitch)`` in radians.

    Returns:
        ``(B,)`` per-sample errors in **degrees**; the caller reduces.
    """
    pred_vec = gaze_to_3d(pred)
    target_vec = gaze_to_3d(target)
    dot = torch.sum(pred_vec * target_vec, dim=-1).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(dot))
