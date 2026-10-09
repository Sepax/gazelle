# Gazelle
Appearance-based gaze estimation from face images, trained on MPIIFaceGaze.

# Overview

_Gazelle_ predicts where a person is looking from a single RGB face image. Given a cropped face, the model regresses a 3D gaze direction, expressed as pitch and yaw angles, without any calibration or special hardware.

The project is trained and evaluated on the MPIIFaceGaze dataset using a leave-one-person-out protocol.

# Setup

The requires [uv](https://docs.astral.sh/uv/) to manage Python and dependencies.

```bash
git clone https://github.com/<your-username>/gazelle.git
cd gazelle
uv sync
```

# Dataset

This project uses [MPIIFaceGaze](https://perceptualui.org/research/datasets/MPIIFaceGaze), which contains face images from 15 participants with 3D gaze annotations. It is hosted on DaRUS (University of Stuttgart), DOI [10.18419/DARUS-3240](https://doi.org/10.18419/DARUS-3240).

The data is released under CC BY-NC-SA 4.0 and may only be used for non-commercial scientific purposes.

## Download

From the repository root:

```bash
uv run python scripts/download.py
```

The download is roughly 1 GB. The archive contains a `readme.txt` and the data itself.

## Expected layout

After extraction, the data folder should look like this:

```
data/
├── MANIFEST.TXT
└── MPIIFaceGaze/
    ├── changelog.txt
    ├── readme.txt
    └── Data/
        ├── p00/
        ├── p01/
        ├── ...
        └── p14/
```

See `readme.txt` in the download for the exact file format and annotation columns. If the folder names differ, set the dataset root in `configs/base.yaml`.

## Citation

If you use this dataset, please cite:

- X. Zhang, Y. Sugano, M. Fritz, A. Bulling. *It's Written All Over Your Face: Full-Face Appearance-Based Gaze Estimation.* CVPRW 2017.
- X. Zhang, Y. Sugano, M. Fritz, A. Bulling. *MPIIGaze: Real-World Dataset and Deep Appearance-Based Gaze Estimation.* IEEE TPAMI 41(1), 2019.

