# Gazelle
Gaze estimation from face images, trained on MPIIFaceGaze.

## Overview

_Gazelle_ predicts where a person is looking from a single RGB face image. Given a cropped face, the model regresses a 3D gaze direction, expressed as pitch and yaw angles, without any calibration or special hardware.

The project is trained and evaluated on the MPIIFaceGaze dataset using a leave-one-person-out protocol.

## Setup

The requires [uv](https://docs.astral.sh/uv/) to manage Python and dependencies.

```bash
git clone https://github.com/sepax/gazelle.git
cd gazelle
uv sync
```

## Dataset
 
This project uses [MPIIFaceGaze](https://perceptualui.org/research/datasets/MPIIFaceGaze), which contains face images from 15 participants with 3D gaze annotations. It is hosted on DaRUS (University of Stuttgart), DOI [10.18419/DARUS-3240](https://doi.org/10.18419/DARUS-3240).
 
The data is released under CC BY-NC-SA 4.0 and may only be used for non-commercial scientific purposes.
 
### Download
 
From the repository root:
 
```bash
uv run python scripts/download.py
```
 
This downloads and extracts MPIIFaceGaze (about 1 GB). It also fetches the 15 small MPIIGaze evaluation sample lists, which define the 3,000 benchmark samples per participant, without downloading the rest of MPIIGaze. Parts that are already present are skipped, and `--force` downloads everything again.
 
After downloading, `data/` looks like this:
 
```
data/
├── MANIFEST.TXT
├── MPIIFaceGaze/
│   ├── changelog.txt
│   ├── readme.txt
│   └── Data/
│       ├── p00/
│       ├── ...
│       └── p14/
└── MPIIGaze/
    └── Evaluation Subset/
        └── sample list for eye image/
            ├── p00.txt
            ├── ...
            └── p14.txt
```
 
See `data/MPIIFaceGaze/readme.txt` for the annotation format.
 
### Preprocessing
 
Faces are normalised to a fixed virtual camera and saved as 224x224 images, following the [GazeHub benchmark](https://phi-ai.buaa.edu.cn/Gazehub/3D-dataset) so that results are comparable with published numbers. The code is the benchmark authors' own, in `src/gazelle/data/gazehub/data_processing_core.py` and `data_processing_mpii.py`. The input and output paths are set at the top of the latter.
 
From the repository root:
 
```bash
uv run python -m gazelle.data.gazehub.data_processing_mpii
```
 
This takes a couple of minutes and writes 3,000 samples per participant (45,000 in total):
 
```
data/processed/MPIIFaceGaze/
├── Image/
│   └── p00/{face,left,right}/1.jpg ... 3000.jpg
└── Label/
    └── p00.label
```
 
Training uses the `Face` image and the `2DGaze` label from each `.label` file, which holds the gaze as (yaw, pitch) in radians. Samples from right eyes are mirrored, together with their labels, as in the benchmark. The eye crops are not used.
 
### Citation
 
- X. Zhang, Y. Sugano, M. Fritz, A. Bulling. *It's Written All Over Your Face: Full-Face Appearance-Based Gaze Estimation.* CVPRW 2017.
- X. Zhang, Y. Sugano, M. Fritz, A. Bulling. *MPIIGaze: Real-World Dataset and Deep Appearance-Based Gaze Estimation.* IEEE TPAMI 41(1), 2019.
- Y. Cheng, H. Wang, Y. Bao, F. Lu. *Appearance-based Gaze Estimation With Deep Learning: A Review and Benchmark.* arXiv:2104.12668, 2021.

## Training

Train a single leave-one-person-out fold:

```bash
uv run python -m gazelle.train --config configs/resnet18.yaml
```

Any `Config` field can be overridden on the command line as `field=value`, so there is no
separate flag per setting:

```bash
uv run python -m gazelle.train --config configs/resnet18.yaml test_subject=3 epochs=10 lr=3e-5
```

Precedence is dataclass defaults, then the YAML file, then overrides. Unknown keys are
rejected rather than ignored. Training runs on [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/),
which picks the accelerator automatically: CUDA, else Apple MPS, else CPU.

Each run writes a timestamped directory:

```
runs/p00_resnet18_1014-1347/
├── config.yaml            every field resolved, defaults and overrides included
├── checkpoints/
│   ├── best.ckpt          lowest validation angular error
│   └── last.ckpt          final epoch, and the one that is scored
└── events.out.tfevents.*  TensorBoard scalars
```

Follow along in a second terminal:

```bash
uv run --no-sync tensorboard --logdir runs
```

### Evaluating a checkpoint

```bash
uv run python -m gazelle.evaluate runs/p00_resnet18_1014-1347/checkpoints/last.ckpt
```

The config stored inside the checkpoint rebuilds the matching model and transform, so a
checkpoint is always scored exactly as it was trained. Pass `--subject N` to score a
participant other than the fold's held-out one.

## Benchmark

The reportable protocol is separate from a single run: train on 14 participants, test on the
15th, repeat for all 15, and report the mean angular error.

```bash
uv run python -m gazelle.benchmark --config configs/resnet18.yaml
```

Use `--folds 0 1 2` to run a subset. Results are grouped per benchmark:

```
benchmarks/1014-1400/
├── summary.json              mean, std, per-fold errors, and the config used
├── p00_resnet18_1014-1400/   each fold is a full run directory
└── ...
```

Hyperparameters must be frozen before running this. Tuning against these numbers overfits the
benchmark and makes them meaningless, so do the tuning on one development fold first, then
freeze and run all 15.

Preprocessing follows the [GazeHub benchmark](https://phi-ai.buaa.edu.cn/Gazehub/3D-dataset),
and the metric is the angle between predicted and true 3D gaze directions, so results are
comparable with published numbers. Training hyperparameters are our own and will differ.

## Testing

Run the test suite:

```bash
uv run pytest
```

Lint and format:

```bash
uv run ruff check .
uv run ruff format .
```
