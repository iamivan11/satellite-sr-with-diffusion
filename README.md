# Satellite Single Image Super-Resolution with Conditional Diffusion Models

## Quickstart

Install dependencies with [uv](https://docs.astral.sh/uv/) from the repo root:

```bash
uv sync
```

Then run the commands below from the `satellite_sr_with_wavelet_diffusion/` directory, with the
environment active (`source .venv/bin/activate`) or each command prefixed with `uv run`.

Pick a config from `config/<dataset>/<arch>/` and point its `datasets.train.dataroot` /
`datasets.val.dataroot` at your data before running. Each split needs `hr/` and `lr/`
subfolders; the shipped configs use Colab `/content/...` paths.

**Train**
```bash
python main.py -c config/sen2venus/wave/train_new_config_4x.json      # add -enable_wandb for W&B
python main.py -c config/sen2venus/wave/train_resume_config_4x.json   # resume from path.resume_state
```

**Evaluate** (PSNR / SSIM / LPIPS / FID)
```bash
python main.py -c config/sen2venus/wave/val_config_4x.json -p val
python main.py -c config/sen2venus/wave/val_config_4x.json -p val --val_upsampler bicubic   # bicubic/lanczos baseline
```

**Inference only** (save SR images, no metrics)
```bash
python inference.py -c config/sen2venus/wave/val_config_4x.json
```

Training writes logs/checkpoints/samples under `experiments/<name>/`; evaluation and inference
write images to the config's `results` path (evaluation also writes `metrics.txt`).

## Datasets

### 1: WorldStrat

- **Description**: A large-scale, open-source dataset featuring nearly 10,000 km² of high-resolution satellite imagery. It ensures a stratified representation of global land-use types and includes under-represented locations like humanitarian sites.
- **Distribution of locations**: Global coverage with focus on humanitarian and conflict-affected regions, including areas in Africa, Middle East, and Asia.
- **Images**: 
    - High Resolution (HR): Sourced from Airbus SPOT 6/7 satellites, providing detailed images at up to 1.5 m/pixel resolution.
    - Low Resolution (LR): Sourced from the public Sentinel-2 satellites, providing multi-spectral images at 10 m/pixel, temporally matched to the HR data.
- **Split**: The official train/validation/test split provided by the dataset creators is used.
    - Train: 3103 (pairs)
    - Val: 393 (pairs)
    - Test (never touched): 394 (pairs)

### 2: SEN2VENUS

- **Description**: An open dataset designed for super-resolution of Sentinel‑2 images by exploiting near-simultaneous acquisitions from the VENµS satellite. It contains cloud-free surface reflectance patches from Sentinel‑2 (10 m and 20 m resolution) paired with spatially-registered high-resolution reference patches (5 m) captured by VENµS on the same day.
- **Distribution of locations**: Covers 29 distinct Earth sites, providing wide geographic diversity.
- **Images**: 
    - High Resolution (HR): VENµS surface reflectance at 5 m resolution, spatially and temporally aligned with Sentinel‑2 data.
    - Low Resolution (LR): Sentinel‑2 surface reflectance at 10 m resolution.
- **Split**: The dataset is organized by location; there is no standardized train/val/test split provided—users must define splits based on their experimental needs.
    - Train: 105,767 (pairs)
    - Val: 13,220 (pairs)
    - Test: 13,220 (pairs)

## Architectures

### 1. SR3

- **Core Concept**: Super-Resolution via Repeated Refinement (SR3) is a conditional diffusion model that progressively denoises from pure noise to reconstruct high-resolution (HR) images guided by a low-resolution (LR) input.
- **Mechanism**: During training, Gaussian noise is added only to the HR target, while the LR input remains clean and serves as conditioning. At inference, the model starts from noise at HR scale and iteratively removes it, each step constrained by the LR image.
- **Advantage**: Produces sharp and diverse HR reconstructions without adversarial training, offering better robustness to distribution shifts compared to GANs or pixel-loss regression models.

### 2. Diffusion Wavelet (DiWa)

- **Core Concept**: DiWa integrates diffusion with the Discrete Wavelet Transform (DWT), modeling super-resolution in the frequency domain rather than in raw pixels.
- **Mechanism**: Condition the model on the LR-upsampled wavelet coefficients (e.g., LL) and denoise a noise tensor in wavelet space to predict missing HF bands (LH/HL/HH); finally IDWT combines predicted details with preserved low-frequency structure to yield the HR image.
- **Advantage**: This approach is efficient as it allows the model to focus its capacity on generating fine details and textures rather than learning the coarse structure of the image, which is already present in the low-resolution input.

## Metrics

### 1: PSNR

- **Description**: Peak Signal-to-Noise Ratio (PSNR) measures the quality of a reconstructed image compared to its original. It is defined via the Mean Squared Error (MSE). A higher PSNR generally indicates a higher quality reconstruction.
- **Range**: 0 to ∞ dB (typically 20-40 dB for natural images). Better when ↑.
- **Formula**:

```math
\text{PSNR}(I_{HR}, I_{SR}) = 10 \cdot \log_{10} \left( \frac{L^2}{\text{MSE}(I_{HR}, I_{SR})} \right)
```

$I_{HR}$ - HR image  
$I_{SR}$ - SR image  
$`L`$ - maximum possible pixel value (e.g., 255 for 8-bit images)  
$\text{MSE}(I_{HR}, I_{SR}) = \frac{1}{HW} \sum_{i=1}^{H} \sum_{j=1}^{W} (I_{HR}(i,j) - I_{SR}(i,j))^2$  
$H, W$ - image height and width

### 2: SSIM

- **Description**: The Structural Similarity Index (SSIM) is a metric that quantifies image quality degradation based on changes in structural information. A full-reference perceptual similarity index that combines three local comparisons—*luminance*, *contrast*, and *structure*—via statistics of means, variances, and cross-covariance.
- **Range**: -1 to 1 (typically 0.1-0.9 for degraded images, 1 = perfect similarity). Most implementations return values in [0, 1], with negatives possible in edge cases depending on data range/regularization constants. Better when ↑.
- **Formula**:

```math
\text{SSIM}(I_{HR}, I_{SR}) =
\left( \frac{2\mu_{I_{HR}} \mu_{I_{SR}} + c_1}{\mu_{I_{HR}}^2 + \mu_{I_{SR}}^2 + c_1} \right)
\left( \frac{2\sigma_{I_{HR}} \sigma_{I_{SR}} + c_2}{\sigma_{I_{HR}}^2 + \sigma_{I_{SR}}^2 + c_2} \right)
\left( \frac{\sigma_{I_{HR} I_{SR}} + c_3}{\sigma_{I_{HR}} \sigma_{I_{SR}} + c_3} \right)
```

$\mu_{I_{HR}}, \mu_{I_{SR}}$ - mean brightness of each image  
$\sigma_{I_{HR}}, \sigma_{I_{SR}}$ - standard deviations (contrast)  
$\sigma_{I_{HR} I_{SR}}$ - covariance (structural similarity)  
$c_1, c_2, c_3$ - small constants for stability  

### 3: LPIPS

- **Description**: The Learned Perceptual Image Patch Similarity (LPIPS) metric computes the distance between the deep features of two images extracted from a pre-trained network (like VGG). It is designed to align better with human perception of image similarity than traditional metrics like PSNR and SSIM.
- **Range**: 0 to 1+ (typically 0.1-0.8 for SR images, 0 = identical images). Better when ↓.
- **Formula**:

```math
\text{LPIPS}(I_{HR}, I_{SR}) = \sum_{l=1}^{L} w_l \cdot \| f_l(I_{HR}) - f_l(I_{SR}) \|_2^2
```

$f_l(I_{HR}), f_l(I_{SR})$ - deep features at layer $l$  
$L$ - number of layers  
$w_l$ - learned weight for layer $l$  

### 4: FID

- **Description**: Frechet Inception Distance (FID) measures the distributional distance between features of real (ground-truth) and generated (SR) images extracted by a pretrained Inception-V3 network (typically the 2048-d pool3 features). It correlates well with perceptual quality at the dataset level.
- **Range**: 0 to ∞ (0 = identical distributions). Better when ↓.
- **Formula**:

```math
\text{FID}(I_{HR}, I_{SR}) = \| \mu_{I_{HR}} - \mu_{I_{SR}} \|_2^2 + d^2(\Sigma_{I_{HR}}, \Sigma_{I_{SR}})
```

$\mu_{I_{HR}}, \mu_{I_{SR}}$ - feature means (real vs generated)  
$\Sigma_{I_{HR}}, \Sigma_{I_{SR}}$ - feature covariances (real vs generated)  
$d^2$ - distance between covariance matrices

### 5: Average Gradient (AG)

- **Description**: Average Gradient (AG) measures the sharpness and detail preservation in reconstructed images by computing the mean magnitude of image gradients. It quantifies how well fine details and edges are preserved during super-resolution, with higher values indicating sharper, more detailed images.
- **Range**: 0 to ∞ (typically 10-100 for natural images). Better when ↑.
- **Formula**:

```math
\text{AG}(I_{SR}) = \frac{1}{HW} \sum_{i=1}^{H} \sum_{j=1}^{W} \sqrt{(\nabla_x I_{SR}(i,j))^2 + (\nabla_y I_{SR}(i,j))^2}
```

$\nabla_x I_{SR}(i,j), \nabla_y I_{SR}(i,j)$ - horizontal and vertical gradients at pixel $(i,j)$   