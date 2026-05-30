"""
Generic dataset processing script

This script:
1. Creates an `lr` folder in each split (e.g., train, val, test)
2. Copies HR images to the LR folders
3. Downscales LR images to a target LR size
4. Optionally downscales HR images to a target HR size

Notes:
- The processing order is important: LR downscaling happens first, then HR downscaling (if enabled).
- Configure defaults in the variables section below or override via CLI arguments.
"""
import os
import shutil
import cv2
import glob
from tqdm import tqdm
from pathlib import Path
import argparse
from typing import Iterable, Tuple, List, Set
import numpy as np


DEFAULT_BASE_DIR: Path = Path("data/SEN2VENUS_3x")
DEFAULT_SPLITS: List[str] = ["train", "val", "test"]
DEFAULT_LR_TARGET_SIZE: Tuple[int, int] = (64, 64)
DEFAULT_HR_TARGET_SIZE: Tuple[int, int] = (192, 192)
DEFAULT_INTERPOLATION: str = "cubic"  # one of: area, linear, cubic, lanczos (bicubic recommended)
DEFAULT_IMAGE_EXTS: Set[str] = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
DEFAULT_PROCESS_HR: bool = False  # If True, also downscale HR images as the last step
DEFAULT_BLUR_MODE: str = "gaussian"  # 'gaussian' or 'mtf'
DEFAULT_MTF_NYQ: float = 0.30  # Typical Sentinel-2 MTF at Nyquist (approx.)


def interpolation_from_name(name: str) -> int:
    name = name.lower()
    if name == "area":
        return cv2.INTER_AREA
    if name == "linear":
        return cv2.INTER_LINEAR
    if name == "cubic":
        return cv2.INTER_CUBIC
    if name == "lanczos":
        return cv2.INTER_LANCZOS4
    return cv2.INTER_AREA


def create_lr_folders(base_dir: Path, splits: Iterable[str]) -> None:
    """Step 1: Create /lr folders in each split"""
    print("\nStep 1: Creating /lr folders...")
    for split in splits:
        lr_dir = Path(base_dir) / split / "lr"
        lr_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Created: {lr_dir}")


def copy_hr_to_lr(base_dir: Path, splits: Iterable[str], image_exts: Set[str]) -> None:
    """Step 2: Copy HR images to LR folders"""
    print("\nStep 2: Copying HR images to LR folders...")
    total_copied = 0
    for split in splits:
        hr_dir = Path(base_dir) / split / "hr"
        lr_dir = Path(base_dir) / split / "lr"

        patterns = [str(hr_dir / f"*{ext}") for ext in image_exts]
        hr_images: List[str] = []
        for pat in patterns:
            hr_images.extend(glob.glob(pat))

        for hr_path in tqdm(hr_images, desc=f"Copying {split} HR->LR"):
            filename = os.path.basename(hr_path)
            lr_path = lr_dir / filename
            shutil.copy2(hr_path, lr_path)
            total_copied += 1
    print(f"  Total images copied: {total_copied}")


def _ensure_odd(kernel_size: int) -> int:
    """Ensure kernel size is odd (required by many blur kernels)."""
    return kernel_size if kernel_size % 2 == 1 else kernel_size + 1


def _apply_gaussian_blur(image: np.ndarray, kernel_size: int, sigma: float) -> np.ndarray:
    """Apply isotropic Gaussian blur with given kernel size and sigma (pixels)."""
    k = _ensure_odd(int(kernel_size))
    return cv2.GaussianBlur(image, (k, k), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REPLICATE)


def _add_awgn(image: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Additive white Gaussian noise on 0-255 intensity scale with stddev sigma."""
    if sigma <= 0:
        return image
    noise = rng.normal(loc=0.0, scale=sigma, size=image.shape).astype(np.float32)
    noisy = image.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def _sigma_from_mtf_nyquist(mtf_nyq: float) -> float:
    """Compute Gaussian sigma (in pixels) whose MTF at Nyquist equals mtf_nyq.

    Uses Gaussian MTF: MTF(f) = exp( -2*pi^2*sigma^2*f^2 ). For Nyquist f=0.5 cycles/pixel.
    Solving for sigma yields: sigma = sqrt(-ln(MTF_nyq)) / pi.
    """
    mtf = float(max(min(mtf_nyq, 0.9999), 1e-6))
    return float(np.sqrt(-np.log(mtf)) / np.pi)


def downsample_lr_images(
    base_dir: Path,
    splits: Iterable[str],
    lr_target_size: Tuple[int, int],
    interpolation_flag: int,
    image_exts: Set[str],
    blur_kernel: int,
    blur_sigma_range: Tuple[float, float],
    noise_sigma_range: Tuple[float, float],
    enable_blur: bool,
    enable_noise: bool,
    rng: np.random.Generator,
    blur_mode: str,
    mtf_nyq: float,
) -> None:
    """Step 3: Create LR by blur -> downsample -> add noise"""
    print(f"\nStep 3: Creating LR with blur -> resize -> noise to {lr_target_size}...")
    total_processed = 0
    for split in splits:
        lr_dir = Path(base_dir) / split / "lr"
        patterns = [str(lr_dir / f"*{ext}") for ext in image_exts]
        lr_images: List[str] = []
        for pat in patterns:
            lr_images.extend(glob.glob(pat))

        for lr_path in tqdm(lr_images, desc=f"Processing {split} LR"):
            img = cv2.imread(lr_path, cv2.IMREAD_COLOR)
            if img is None:
                print(f"  ERROR: Could not load {lr_path}")
                continue

            # 1) Blur
            if enable_blur:
                if blur_mode == "mtf":
                    sigma_blur = _sigma_from_mtf_nyquist(mtf_nyq)
                    k_est = max(3, int(np.round(6.0 * sigma_blur)))
                    k_use = _ensure_odd(k_est)
                    img = _apply_gaussian_blur(img, k_use, sigma_blur)
                else:
                    sigma_blur = float(rng.uniform(blur_sigma_range[0], blur_sigma_range[1]))
                    img = _apply_gaussian_blur(img, blur_kernel, sigma_blur)

            # 2) Downsample (bicubic or as chosen)
            downsampled = cv2.resize(img, lr_target_size, interpolation=interpolation_flag)

            # 3) Add noise
            if enable_noise:
                sigma_noise = float(rng.uniform(noise_sigma_range[0], noise_sigma_range[1]))
                downsampled = _add_awgn(downsampled, sigma_noise, rng)

            cv2.imwrite(lr_path, downsampled)
            total_processed += 1
    print(f"  Total LR images processed: {total_processed}")


def downsample_hr_images(base_dir: Path, splits: Iterable[str], hr_target_size: Tuple[int, int], interpolation_flag: int, image_exts: Set[str]) -> None:
    """Step 4: Downsample HR images to the target HR size"""
    print(f"\nStep 4: Downsampling HR images to {hr_target_size}...")
    total_processed = 0
    for split in splits:
        hr_dir = Path(base_dir) / split / "hr"
        patterns = [str(hr_dir / f"*{ext}") for ext in image_exts]
        hr_images: List[str] = []
        for pat in patterns:
            hr_images.extend(glob.glob(pat))

        for hr_path in tqdm(hr_images, desc=f"Downsampling {split} HR"):
            img = cv2.imread(hr_path, cv2.IMREAD_COLOR)
            if img is None:
                print(f"  ERROR: Could not load {hr_path}")
                continue
            downsampled = cv2.resize(img, hr_target_size, interpolation=interpolation_flag)
            cv2.imwrite(hr_path, downsampled)
            total_processed += 1
    print(f"  Total HR images downsampled: {total_processed}")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare dataset LR/HR folders: copy HR->LR, downscale LR, optional HR downscale"
    )
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR, help="Dataset base directory")
    parser.add_argument("--splits", type=str, nargs="*", default=DEFAULT_SPLITS, help="Splits to process")
    parser.add_argument("--lr-size", type=int, nargs=2, metavar=("W", "H"), default=list(DEFAULT_LR_TARGET_SIZE), help="Target LR size (W H)")
    parser.add_argument("--hr-size", type=int, nargs=2, metavar=("W", "H"), default=list(DEFAULT_HR_TARGET_SIZE), help="Target HR size (W H)")
    parser.add_argument("--interpolation", choices=["area", "linear", "cubic", "lanczos"], default=DEFAULT_INTERPOLATION, help="Interpolation method")
    parser.add_argument("--process-hr", action="store_true", default=DEFAULT_PROCESS_HR, help="Also downscale HR images to target HR size")
    # Degradation controls
    parser.add_argument("--blur-mode", choices=["gaussian", "mtf"], default=DEFAULT_BLUR_MODE, help="Blur model: isotropic Gaussian or MTF-based Gaussian")
    parser.add_argument("--mtf-nyq", type=float, default=DEFAULT_MTF_NYQ, help="MTF at Nyquist (0.5 cyc/px) for MTF-based blur. E.g., 0.28-0.35 for Sentinel-2")
    parser.add_argument("--blur-kernel", type=int, default=7, help="Gaussian blur kernel size (odd)")
    parser.add_argument("--blur-sigma", type=float, nargs=2, metavar=("MIN", "MAX"), default=[1.5, 2.0], help="Range to sample Gaussian blur sigma from (in pixels) for 'gaussian' mode, or ignored if --blur-mode mtf")
    parser.add_argument("--noise-sigma", type=float, nargs=2, metavar=("MIN", "MAX"), default=[0.0, 10.0], help="Range to sample AWGN sigma from (0-255 scale)")
    parser.add_argument("--no-blur", action="store_true", help="Disable blur step for LR creation")
    parser.add_argument("--no-noise", action="store_true", help="Disable noise step for LR creation")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for degradation sampling")
    return parser.parse_args()


def main():
    args = parse_args()

    base_dir = args.base_dir
    splits = args.splits
    lr_size = (int(args.lr_size[0]), int(args.lr_size[1]))
    hr_size = (int(args.hr_size[0]), int(args.hr_size[1]))
    interp_flag = interpolation_from_name(args.interpolation)
    image_exts = DEFAULT_IMAGE_EXTS
    rng = np.random.default_rng(args.seed)

    print("Dataset Processing")
    print("=" * 50)
    print(f"Base directory: {base_dir}")
    print(f"Splits: {', '.join(splits)}")
    print(f"LR target size: {lr_size}")
    print(f"HR target size: {hr_size}")
    print(f"Interpolation: {args.interpolation}")
    print("=" * 50)

    # Step 1: Create LR folders
    create_lr_folders(base_dir, splits)

    # Step 2: Copy HR images to LR folders
    copy_hr_to_lr(base_dir, splits, image_exts)

    # Step 3: Create LR with blur -> resize -> noise
    downsample_lr_images(
        base_dir=base_dir,
        splits=splits,
        lr_target_size=lr_size,
        interpolation_flag=interp_flag,
        image_exts=image_exts,
        blur_kernel=args.blur_kernel,
        blur_sigma_range=(float(args.blur_sigma[0]), float(args.blur_sigma[1])),
        noise_sigma_range=(float(args.noise_sigma[0]), float(args.noise_sigma[1])),
        enable_blur=(not args.no_blur),
        enable_noise=(not args.no_noise),
        rng=rng,
        blur_mode=args.blur_mode,
        mtf_nyq=float(args.mtf_nyq),
    )

    # Optional: Step 4 HR downsample
    if args.process_hr:
        downsample_hr_images(base_dir, splits, hr_size, interp_flag, image_exts)

    print("\n" + "=" * 50)
    print("SUCCESS: All processing completed!")
    print("=" * 50)

if __name__ == "__main__":
    main()