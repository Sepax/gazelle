from pathlib import Path

import pytest
from PIL import Image

from gazelle.data.transforms import NATIVE_SIZE

_HEADER = (
    "Face Left Right Origin WhichEye 3DGaze 3DHead 2DGaze 2DHead Rmat Smat GazeOrigin"
)


def gaze_for(subject: int, index: int) -> tuple[float, float]:
    """Distinctive (yaw, pitch) per sample, asymmetric so a swap is detectable."""
    return (0.1 * subject + 0.01 * index, -0.5 - 0.01 * index)


def make_data_root(
    root: Path, subjects: list[int], n_samples: int = 4, write_images: bool = True
) -> Path:
    """Write a miniature copy of the processed MPIIFaceGaze layout.

    Mirrors the real tree: ``Label/pNN.label`` with a header row and twelve
    whitespace-separated columns, plus ``Image/pNN/face/K.jpg``.
    """
    for subject in subjects:
        rows = [_HEADER]
        for i in range(1, n_samples + 1):
            face = f"p{subject:02d}/face/{i}.jpg"
            yaw, pitch = gaze_for(subject, i)
            rows.append(
                " ".join(
                    [
                        face,
                        f"p{subject:02d}/left/{i}.jpg",
                        f"p{subject:02d}/right/{i}.jpg",
                        "day01/0001.jpg",
                        "left",
                        "0.0,0.0,-1.0",
                        "0.0,0.0,-1.0",
                        f"{yaw},{pitch}",
                        "0.0,0.0",
                        "0.0,0.0,0.0",
                        "1.0,1.0,1.0",
                        "0.0,0.0,600.0",
                    ]
                )
            )
            if write_images:
                img_path = root / "Image" / face
                img_path.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (NATIVE_SIZE, NATIVE_SIZE), (i * 10, 20, 30)).save(img_path)

        label_path = root / "Label" / f"p{subject:02d}.label"
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text("\n".join(rows) + "\n")
    return root


@pytest.fixture
def data_root(tmp_path: Path) -> Path:
    """Three participants, four samples each."""
    return make_data_root(tmp_path, subjects=[0, 1, 2], n_samples=4)
