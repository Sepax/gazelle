from collections.abc import Callable
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

# Column indices in the GazeHub .label files.
_FACE = 0
_GAZE_2D = 7


class GazeDataset(Dataset):
    """Normalised MPIIFaceGaze face crops with their 2D gaze labels.

    Reads the ``.label`` files written by the GazeHub preprocessing, using only
    the ``Face`` and ``2DGaze`` columns. Eye crops are ignored.
    """

    def __init__(
        self,
        data_root: str | Path,
        subjects: list[int],
        transform: Callable[[Image.Image], torch.Tensor],
    ):
        """Index the samples for the given participants.

        Args:
            data_root: Root holding ``Image/`` and ``Label/``.
            subjects: Participant IDs to include, e.g. ``Split.train``.
            transform: Maps a PIL image to a normalised tensor; train and
                eval pipelines differ.

        Raises:
            ValueError: If ``data_root`` is not a directory.
            FileNotFoundError: If a participant's label file is missing.
        """
        self.root = Path(data_root)
        if not self.root.is_dir():
            raise ValueError(f"Data root '{self.root}' is invalid")

        self.subjects = list(subjects)
        self.transform = transform
        self.samples = self._collect_samples()

    def _collect_samples(self) -> list[tuple[str, tuple[float, float]]]:
        """Pair each face image path with its ``(yaw, pitch)`` label."""
        samples = []
        for subject in self.subjects:
            label_file = self.root / "Label" / f"p{subject:02d}.label"
            if not label_file.is_file():
                raise FileNotFoundError(f"Missing label file: {label_file}")

            with label_file.open() as f:
                next(f)  # header row
                for line in f:
                    parts = line.split()
                    if not parts:
                        continue
                    yaw, pitch = parts[_GAZE_2D].split(",")
                    samples.append((parts[_FACE], (float(yaw), float(pitch))))
        return samples

    def __len__(self) -> int:
        """Total samples across the included participants, 3000 each."""
        return len(self.samples)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Load one sample.

        Args:
            i: Index into the flattened sample list.

        Returns:
            ``(image, gaze)`` where image is whatever ``transform`` produces
            from the RGB crop, and gaze is float32 ``(2,)`` holding
            ``(yaw, pitch)`` in radians.
        """
        path, gaze = self.samples[i]
        with Image.open(self.root / "Image" / path) as img:
            image = self.transform(img.convert("RGB"))
        return image, torch.tensor(gaze, dtype=torch.float32)
