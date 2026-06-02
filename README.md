# Satellite Single Image Super-Resolution with Conditional Diffusion Models

This project studies single-image super-resolution of satellite imagery with
conditional diffusion models. It compares the wavelet-domain Diffusion-Wavelet
(DiWa) model against the pixel-space SR3 DDPM and interpolation baselines,
trained and evaluated on the WorldStrat and SEN2VENµS datasets across 6.6×, 4×,
3×, and 2× scales. It also quantifies how training on synthetically degraded
versus real low-resolution images affects reconstruction quality. Everything can
be found in the [paper](report/main.pdf) that comes with the project.

## Datasets

### 1: WorldStrat

- **Description**: A large-scale, open-source dataset featuring nearly 10,000
  km² of high-resolution satellite imagery. It ensures a stratified
  representation of global land-use types and includes under-represented
  locations like humanitarian sites.
- **Distribution of locations**: Global coverage with focus on humanitarian and
  conflict-affected regions, including areas in Africa, Middle East, and Asia.
- **Images**:
  - High Resolution (HR): Sourced from Airbus SPOT 6/7 satellites, providing
    detailed images at up to 1.5 m/pixel resolution.
  - Low Resolution (LR): Sourced from the public Sentinel-2 satellites,
    providing multi-spectral images at 10 m/pixel, temporally matched to the HR
    data.
- **Split**: The official train/validation/test split provided by the dataset
  creators is used (in pairs).
  - Train: 3103 (80%)
  - Val: 393 (10%)
  - Test (never touched): 394 (10%)

### 2: SEN2VENµS

- **Description**: An open dataset designed for super-resolution of Sentinel‑2
  images by exploiting near-simultaneous acquisitions from the VENµS satellite.
  It contains cloud-free surface reflectance patches from Sentinel‑2 (10 m and
  20 m resolution) paired with spatially-registered high-resolution reference
  patches (5 m) captured by VENµS on the same day.
- **Distribution of locations**: Covers 29 distinct Earth sites, providing wide
  geographic diversity.
- **Images**:
  - High Resolution (HR): VENµS surface reflectance at 5 m resolution, spatially
    and temporally aligned with Sentinel‑2 data.
  - Low Resolution (LR): Sentinel‑2 surface reflectance at 10 m resolution.
- **Split**: The dataset is organized by location; there is no standardized
  train/val/test split provided—users must define splits based on their
  experimental needs (in pairs).
  - Train: 105,767 (80%)
  - Val: 13,220 (10%)
  - Test: 13,220 (10%)

## Architectures

### 1. SR3

- **Core Concept**: Super-Resolution via Repeated Refinement (SR3) is a
  conditional diffusion model that progressively denoises from pure noise to
  reconstruct high-resolution (HR) images guided by a low-resolution (LR) input.
- **Mechanism**: During training, Gaussian noise is added only to the HR target,
  while the LR input remains clean and serves as conditioning. At inference, the
  model starts from noise at HR scale and iteratively removes it, each step
  constrained by the LR image.
- **Advantage**: Produces sharp and diverse HR reconstructions without
  adversarial training, offering better robustness to distribution shifts
  compared to GANs or pixel-loss regression models.

### 2. Diffusion Wavelet (DiWa)

- **Core Concept**: DiWa integrates diffusion with the Discrete Wavelet
  Transform (DWT), modeling super-resolution in the frequency domain rather than
  in raw pixels.
- **Mechanism**: Condition the model on the LR-upsampled wavelet coefficients
  (e.g., LL) and denoise a noise tensor in wavelet space to predict missing HF
  bands (LH/HL/HH); finally IDWT combines predicted details with preserved
  low-frequency structure to yield the HR image.
- **Advantage**: This approach is efficient as it allows the model to focus its
  capacity on generating fine details and textures rather than learning the
  coarse structure of the image, which is already present in the low-resolution
  input.

## Metrics

Reconstructions are evaluated with five standard super-resolution metrics.

| Metric    | Measures                                                       | Better |
| --------- | -------------------------------------------------------------- | :----: |
| **PSNR**  | Pixel-level fidelity, derived from the mean squared error      |   ↑    |
| **SSIM**  | Structural similarity (luminance, contrast, structure)         |   ↑    |
| **LPIPS** | Perceptual similarity between deep-network features            |   ↓    |
| **FID**   | Realism of the SR feature distribution vs. real HR (Inception) |   ↓    |
| **AG**    | Sharpness / edge clarity, from average gradient magnitude      |   ↑    |

## Setup

### Requirements

- Python 3.10–3.12
- [uv](https://docs.astral.sh/uv/) package manager

### Installation

```bash
uv sync                                 # create the environment and install deps
source .venv/bin/activate               # or prefix each command below with `uv run`
cd satellite_sr_with_diffusion          # run all commands from here
```

## Train

Runs are configured with [Hydra](https://hydra.cc): compose a run by selecting a
**dataset** (`data=`) and an **architecture** (`model=`), and override any field
on the command line. Point the config at your data first
(`datasets.train.dataroot` / `datasets.val.dataroot`); each split is a folder
with `hr/` and `lr/` subfolders.

```bash
python main.py data=sen2venus_4x model=wave                      # train (add wandb.enable=true for W&B)
python main.py data=sen2venus_4x model=wave phase=val \
    path.resume_state=/path/to/checkpoint                        # evaluate (PSNR/SSIM/LPIPS/FID)
python main.py data=sen2venus_4x model=wave phase=val \
    val_upsampler=bicubic path.resume_state=/path/to/checkpoint  # bicubic/lanczos baseline
```

`data` ∈ {`sen2venus_4x`, `sen2venus_3x`, `sen2venus_2x`, `worldstrat`,
`custom`} and `model` ∈ {`wave`, `sr3`}. Training writes to
`experiments/<name>/`; evaluation and inference write to `results/<name>/`,
where `<name>` defaults to `<dataset>_<scale>_<arch>`.

### Use your own dataset / resolution

The pipeline is dataset-agnostic: any dataset works as long as each split is a
directory with `hr/` and `lr/` subfolders holding the **same number of
index-aligned images** (the _i_-th LR image is the low-res counterpart of the
_i_-th HR image). Each `lr/` image must be **pre-upsampled to the HR size**, so
`hr/` and `lr/` images share the same dimensions (use
`python scripts/prepare_data.py upscale` to prepare them). Resolution is
controlled by a single knob, `hr_size` (the HR patch size, a multiple of 32);
the model's `image_size` and attention resolutions are derived from it
automatically (the wavelet model operates at `hr_size/2`). Either edit
[configs/data/custom.yaml](configs/data/custom.yaml) or override on the command
line:

```bash
python main.py data=custom model=wave hr_size=128 \
    datasets.train.dataroot=/path/to/your_dataset/train \
    datasets.val.dataroot=/path/to/your_dataset/val
```

To add a dataset permanently, copy a file in [configs/data/](configs/data/), set
its `hr_size` and dataroots, and select it with `data=<your_file>`. See
[configs/config.yaml](configs/config.yaml) for all overridable fields.

## Inference

```bash
python inference.py data=sen2venus_4x model=wave \
    path.resume_state=/path/to/checkpoint                        # inference only (save SR images)
```

## Structure

```
.
├── configs/                       # Hydra hierarchical configs
│   ├── config.yaml                #   root config (defaults, train/path/wandb settings)
│   ├── data/                      #   dataset + resolution groups (sen2venus_*, worldstrat, custom)
│   └── model/                     #   architecture groups (wave, sr3)
├── satellite_sr_with_diffusion/   # main package
│   ├── main.py                    # entry point: training & evaluation (phase=train / phase=val)
│   ├── inference.py               # entry point: inference only (save SR images, no metrics)
│   ├── core/                      # config loader, logging, metrics, shared inference engine
│   ├── data/                      # generic HR/LR pair loader (image_pair_dataset.py) & helpers
│   ├── model/                     # diffusion models
│   │   ├── sr3_modules/           #   SR3   — pixel-space U-Net + diffusion
│   │   └── wave_modules/          #   DiWa  — wavelet-space U-Net + diffusion
│   └── analysis/                  # texture/sharpness analysis (ag.py, glcm.py, lbp.py)
├── scripts/                       # data preparation
│   ├── prepare_data.py            #   degrade / resize / upscale subcommands
│   ├── match_lr_colors_to_hr.py   #   histogram-match LR colours to HR
│   ├── rename_images.py           #   batch rename
│   └── *_split.csv, worldstrat_config.json   # dataset splits + cloud labels
├── report/main.pdf                # compiled project report (LaTeX sources are gitignored)
├── pyproject.toml                 # dependencies + ruff/codespell config (uv-managed)
└── uv.lock                        # pinned dependency lockfile
```

## Tech Stack

The stack favors reproducibility and frequency-domain super-resolution — each
concern (training, configuration, metrics, tracking) handled by a focused,
GPU-friendly tool.

| Layer               | Choice                                        | Why this over alternatives                                                                                                                  |
| ------------------- | --------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| Deep learning       | **PyTorch + Torchvision**                     | Dynamic graphs make it well suited to research and rapid iteration; mature ecosystem for diffusion models, with transforms and backbones    |
| Wavelet transforms  | **PyTorch Wavelets**                          | GPU, autograd-ready 2D DWT/IDWT for the DiWa frequency-domain model (vs CPU PyWavelets)                                                     |
| Image processing    | **Pillow · OpenCV (headless) · scikit-image** | Fast I/O, resizing and degradation pipelines (headless OpenCV drops GUI deps on servers); scikit-image powers the GLCM/LBP texture analysis |
| Metrics             | **torchmetrics[image] · lpips**               | Batched, on-device PSNR / SSIM / LPIPS / FID (vs hand-rolled metrics)                                                                       |
| Experiment tracking | **Weights & Biases**                          | Hosted dashboards for losses, samples and checkpoints across many runs                                                                      |
| Config              | **Hydra**                                     | Composable YAML groups + CLI overrides → swap dataset/model/scale in one flag (vs argparse)                                                 |
| Env & packaging     | **uv**                                        | Rust-fast installs with a real lockfile                                                                                                     |
| Lint & format       | **Ruff**                                      | One Rust tool replaces flake8 + black + isort, near-instant                                                                                 |
| Pre-commit hooks    | **pre-commit** (+ prettier, codespell)        | Auto-enforces formatting, typo and large-file checks before every commit                                                                    |

## Hardware

Training spanned whichever accelerators were available at the time, so timely
access — not just raw throughput — shaped the hardware choice for each stage.

| Environment    | Device                   | Role                                     |
| -------------- | ------------------------ | ---------------------------------------- |
| Local          | **MacBook Pro M3 (MPS)** | Development, debugging, small-scale runs |
| Google Colab   | **NVIDIA Tesla T4**      | Mid-scale training and evaluation        |
| BlueBEAR (HPC) | **NVIDIA A100 40 GB**    | Large-scale training of the final models |
