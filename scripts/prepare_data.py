"""
Dataset preparation utilities (unified).

Subcommands:
  degrade   Build LR from HR: create lr/ folders, copy HR->LR, then
            blur -> downsample -> noise; optionally downscale HR too.
  resize    Resize HR and LR images in-place to target sizes.
  upscale   Bicubically upsample LR images to a target edge size.

Run `python scripts/prepare_data.py <subcommand> --help` for options.
"""

import argparse
import glob
import os
import shutil
from collections.abc import Iterable
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

IMAGE_EXTS: set[str] = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
UPSAMPLE_EXTS: set[str] = {".jpg", ".jpeg", ".png"}


# ===========================================================================
# degrade  (build LR from HR)
# ===========================================================================
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


def copy_hr_to_lr(base_dir: Path, splits: Iterable[str], image_exts: set[str]) -> None:
    """Step 2: Copy HR images to LR folders"""
    print("\nStep 2: Copying HR images to LR folders...")
    total_copied = 0
    for split in splits:
        hr_dir = Path(base_dir) / split / "hr"
        lr_dir = Path(base_dir) / split / "lr"

        patterns = [str(hr_dir / f"*{ext}") for ext in image_exts]
        hr_images: list[str] = []
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
    return cv2.GaussianBlur(
        image, (k, k), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REPLICATE
    )


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
    lr_target_size: tuple[int, int],
    interpolation_flag: int,
    image_exts: set[str],
    blur_kernel: int,
    blur_sigma_range: tuple[float, float],
    noise_sigma_range: tuple[float, float],
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
        lr_images: list[str] = []
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


def downsample_hr_images(
    base_dir: Path,
    splits: Iterable[str],
    hr_target_size: tuple[int, int],
    interpolation_flag: int,
    image_exts: set[str],
) -> None:
    """Step 4: Downsample HR images to the target HR size"""
    print(f"\nStep 4: Downsampling HR images to {hr_target_size}...")
    total_processed = 0
    for split in splits:
        hr_dir = Path(base_dir) / split / "hr"
        patterns = [str(hr_dir / f"*{ext}") for ext in image_exts]
        hr_images: list[str] = []
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


def run_degrade(args: argparse.Namespace) -> None:
    base_dir = args.base_dir
    splits = args.splits
    lr_size = (int(args.lr_size[0]), int(args.lr_size[1]))
    hr_size = (int(args.hr_size[0]), int(args.hr_size[1]))
    interp_flag = interpolation_from_name(args.interpolation)
    image_exts = IMAGE_EXTS
    rng = np.random.default_rng(args.seed)

    print("Dataset Processing")
    print("=" * 50)
    print(f"Base directory: {base_dir}")
    print(f"Splits: {', '.join(splits)}")
    print(f"LR target size: {lr_size}")
    print(f"HR target size: {hr_size}")
    print(f"Interpolation: {args.interpolation}")
    print("=" * 50)

    create_lr_folders(base_dir, splits)
    copy_hr_to_lr(base_dir, splits, image_exts)
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
    if args.process_hr:
        downsample_hr_images(base_dir, splits, hr_size, interp_flag, image_exts)

    print("\n" + "=" * 50)
    print("SUCCESS: All processing completed!")
    print("=" * 50)


# ===========================================================================
# resize  (resize HR/LR in place)
# ===========================================================================
def resize_images_in_directory(
    directory: Path, target_size: tuple[int, int], description: str
) -> int:
    """Resize all images in a directory to target_size (in place)."""
    if not directory.exists():
        print(f"Directory not found: {directory}")
        return 0

    image_files = [f for f in os.listdir(directory) if Path(f).suffix.lower() in IMAGE_EXTS]
    if not image_files:
        print(f"No images found in {directory}")
        return 0

    processed = 0
    errors = 0
    print(f"\nProcessing {len(image_files)} images in {directory}")

    for filename in tqdm(image_files, desc=description):
        try:
            file_path = directory / filename
            img = cv2.imread(str(file_path))
            if img is None:
                print(f"Could not read image: {filename}")
                errors += 1
                continue
            # Skip if already correct size
            if img.shape[1] == target_size[0] and img.shape[0] == target_size[1]:
                continue
            # INTER_AREA is good for downsampling
            resized_img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(file_path), resized_img)
            processed += 1
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            errors += 1

    print(
        f"Resized {processed} images, {errors} errors, "
        f"{len(image_files) - processed - errors} already correct size"
    )
    return processed


def run_resize(args: argparse.Namespace) -> None:
    base_dir = args.base_dir
    splits = args.splits
    hr_size = (int(args.hr_size[0]), int(args.hr_size[1]))
    lr_size = (int(args.lr_size[0]), int(args.lr_size[1]))

    hr_directories = [base_dir / split / "hr" for split in splits]
    lr_directories = [base_dir / split / "lr" for split in splits]

    if args.dry_run:
        print("DRY RUN MODE - No images will be modified")
        for hr_dir in hr_directories:
            if hr_dir.exists():
                count = len([f for f in os.listdir(hr_dir) if Path(f).suffix.lower() in IMAGE_EXTS])
                print(f"Would resize {count} HR images in {hr_dir} to {hr_size}")
            else:
                print(f"HR directory not found: {hr_dir}")
        for lr_dir in lr_directories:
            if lr_dir.exists():
                count = len([f for f in os.listdir(lr_dir) if Path(f).suffix.lower() in IMAGE_EXTS])
                print(f"Would resize {count} LR images in {lr_dir} to {lr_size}")
            else:
                print(f"LR directory not found: {lr_dir}")
        return

    print("=== RESIZING DATASET IMAGES ===")
    print(f"Target sizes: HR={hr_size}, LR={lr_size}")
    total_processed = 0

    print(f"\n--- Processing HR Images ({hr_size[0]}x{hr_size[1]}) ---")
    for hr_dir in hr_directories:
        total_processed += resize_images_in_directory(hr_dir, hr_size, f"HR {hr_dir.parent.name}")

    print(f"\n--- Processing LR Images ({lr_size[0]}x{lr_size[1]}) ---")
    for lr_dir in lr_directories:
        total_processed += resize_images_in_directory(lr_dir, lr_size, f"LR {lr_dir.parent.name}")

    print("\n=== SUMMARY ===")
    print(f"Total images resized: {total_processed}")
    print("Resize operation completed!")


# ===========================================================================
# upscale  (bicubic upsample LR)
# ===========================================================================
def upsample_images_in_directory(directory: Path, target_size: tuple[int, int]) -> None:
    """Bicubically upsample all images in a directory to target_size (in place)."""
    if not directory.is_dir():
        print(f"Directory not found: {directory}, skipping.")
        return

    image_files = sorted(
        f for f in os.listdir(directory) if Path(f).suffix.lower() in UPSAMPLE_EXTS
    )
    if not image_files:
        print(f"No images found in {directory}.")
        return

    print(
        f"\nUpsampling {len(image_files)} images in '{directory}' "
        f"to {target_size[0]}x{target_size[1]}..."
    )

    for filename in tqdm(image_files, desc=f"Upsampling {directory.parent.name}/{directory.name}"):
        try:
            image_path = directory / filename
            with Image.open(image_path) as img:
                if img.mode not in ["RGB"]:
                    img = img.convert("RGB")
                upsampled_img = img.resize(target_size, Image.BICUBIC)
                upsampled_img.save(image_path, "JPEG", quality=100)
        except Exception as e:
            print(f"Error processing {filename}: {e}")


def run_upscale(args: argparse.Namespace) -> None:
    print("--- WARNING ---")
    print("This will OVERWRITE low-resolution (lr) images.")
    print("There is no undo. Consider backing up your data.")

    size = (args.target_size, args.target_size)
    print("\n=== UPSAMPLING LR IMAGES ===")

    lr_directories = [
        args.base_dir / "train" / "lr",
        args.base_dir / "test" / "lr",
        args.base_dir / "val" / "lr",
    ]
    for directory in lr_directories:
        upsample_images_in_directory(directory, size)

    print("\n=== All upsampling operations completed successfully! ===")


# ===========================================================================
# CLI
# ===========================================================================
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dataset preparation utilities.")
    sub = parser.add_subparsers(dest="command", required=True)

    # --- degrade ---
    d = sub.add_parser("degrade", help="Build LR from HR (copy, blur -> downsample -> noise).")
    d.add_argument(
        "--base-dir", type=Path, default=Path("data/SEN2VENUS_3x"), help="Dataset base directory"
    )
    d.add_argument(
        "--splits", type=str, nargs="*", default=["train", "val", "test"], help="Splits to process"
    )
    d.add_argument(
        "--lr-size",
        type=int,
        nargs=2,
        metavar=("W", "H"),
        default=[64, 64],
        help="Target LR size (W H)",
    )
    d.add_argument(
        "--hr-size",
        type=int,
        nargs=2,
        metavar=("W", "H"),
        default=[192, 192],
        help="Target HR size (W H)",
    )
    d.add_argument(
        "--interpolation",
        choices=["area", "linear", "cubic", "lanczos"],
        default="cubic",
        help="Interpolation method",
    )
    d.add_argument(
        "--process-hr",
        action="store_true",
        default=False,
        help="Also downscale HR images to target HR size",
    )
    d.add_argument(
        "--blur-mode",
        choices=["gaussian", "mtf"],
        default="gaussian",
        help="Blur model: isotropic Gaussian or MTF-based Gaussian",
    )
    d.add_argument(
        "--mtf-nyq",
        type=float,
        default=0.30,
        help="MTF at Nyquist (0.5 cyc/px) for MTF blur, e.g. 0.28-0.35 for Sentinel-2",
    )
    d.add_argument("--blur-kernel", type=int, default=7, help="Gaussian blur kernel size (odd)")
    d.add_argument(
        "--blur-sigma",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=[1.5, 2.0],
        help="Gaussian blur sigma range (pixels); ignored if --blur-mode mtf",
    )
    d.add_argument(
        "--noise-sigma",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=[0.0, 10.0],
        help="AWGN sigma range (0-255 scale)",
    )
    d.add_argument("--no-blur", action="store_true", help="Disable blur step")
    d.add_argument("--no-noise", action="store_true", help="Disable noise step")
    d.add_argument("--seed", type=int, default=0, help="Random seed for degradation sampling")
    d.set_defaults(func=run_degrade)

    # --- resize ---
    r = sub.add_parser("resize", help="Resize HR and LR images in-place to target sizes.")
    r.add_argument(
        "--base-dir", type=Path, default=Path("data/DATASET"), help="Base dataset directory"
    )
    r.add_argument(
        "--splits", type=str, nargs="*", default=["train", "test", "val"], help="Splits to process"
    )
    r.add_argument(
        "--hr-size",
        type=int,
        nargs=2,
        metavar=("W", "H"),
        default=[192, 192],
        help="HR target size (W H)",
    )
    r.add_argument(
        "--lr-size",
        type=int,
        nargs=2,
        metavar=("W", "H"),
        default=[48, 48],
        help="LR target size (W H)",
    )
    r.add_argument(
        "--dry-run", action="store_true", help="Show what would be done without resizing"
    )
    r.set_defaults(func=run_resize)

    # --- upscale ---
    u = sub.add_parser("upscale", help="Bicubically upsample LR images to a target edge size.")
    u.add_argument(
        "--base-dir",
        type=Path,
        default=Path("data/DATASET"),
        help="Path to the dataset base directory",
    )
    u.add_argument(
        "--target-size", type=int, default=192, help="Target edge size (e.g. 192 for 192x192)"
    )
    u.set_defaults(func=run_upscale)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
