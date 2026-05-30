"""
Resize all dataset images to standardized resolutions:
- HR images (train/test/val/hr) -> 192x192 pixels (default)
- LR images (train/test/val/lr) -> 48x48 pixels (default)

Processes images in-place, preserving filenames and directory structure.
Configure defaults at the top or override via CLI arguments.
"""

import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
from typing import Tuple, Set


DEFAULT_BASE_DIR: Path = Path("data/DATASET")
DEFAULT_SPLITS = ["train", "test", "val"]
DEFAULT_HR_SIZE: Tuple[int, int] = (192, 192)
DEFAULT_LR_SIZE: Tuple[int, int] = (48, 48)
DEFAULT_IMAGE_EXTS: Set[str] = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}


def resize_images_in_directory(directory: Path, target_size: Tuple[int, int], description: str) -> int:
    """
    Resize all images in a directory to target_size.
    
    Args:
        directory (Path): Directory containing images
        target_size (tuple): (width, height) target size
        description (str): Description for progress bar
    """
    if not directory.exists():
        print(f"Directory not found: {directory}")
        return 0
    
    # Get all image files
    image_files = [f for f in os.listdir(directory)
                   if Path(f).suffix.lower() in DEFAULT_IMAGE_EXTS]
    
    if not image_files:
        print(f"No images found in {directory}")
        return 0
    
    processed = 0
    errors = 0
    
    print(f"\nProcessing {len(image_files)} images in {directory}")
    
    for filename in tqdm(image_files, desc=description):
        try:
            file_path = directory / filename
            
            # Read image with OpenCV
            img = cv2.imread(str(file_path))
            if img is None:
                print(f"Could not read image: {filename}")
                errors += 1
                continue
            
            # Check if resize is needed
            if img.shape[1] == target_size[0] and img.shape[0] == target_size[1]:
                continue  # Skip if already correct size
            
            # Resize image using INTER_AREA (good for downsampling)
            resized_img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
            
            # Save back to same location
            cv2.imwrite(str(file_path), resized_img)
            processed += 1
                
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            errors += 1
    
    print(f"Resized {processed} images, {errors} errors, {len(image_files) - processed - errors} already correct size")
    return processed


def main():
    parser = argparse.ArgumentParser(description='Resize dataset images to standard resolutions')
    parser.add_argument('--base-dir', type=Path, default=DEFAULT_BASE_DIR, help='Base dataset directory')
    parser.add_argument('--splits', type=str, nargs='*', default=DEFAULT_SPLITS, help='Splits to process')
    parser.add_argument('--hr-size', type=int, nargs=2, metavar=('W', 'H'), default=list(DEFAULT_HR_SIZE), help='HR target size (W H)')
    parser.add_argument('--lr-size', type=int, nargs=2, metavar=('W', 'H'), default=list(DEFAULT_LR_SIZE), help='LR target size (W H)')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done without actually resizing')
    args = parser.parse_args()

    base_dir = args.base_dir
    splits = args.splits
    hr_size = (int(args.hr_size[0]), int(args.hr_size[1]))
    lr_size = (int(args.lr_size[0]), int(args.lr_size[1]))

    # Define directories to process
    hr_directories = [base_dir / split / 'hr' for split in splits]
    lr_directories = [base_dir / split / 'lr' for split in splits]

    if args.dry_run:
        print("DRY RUN MODE - No images will be modified")
        for hr_dir in hr_directories:
            if hr_dir.exists():
                count = len([f for f in os.listdir(hr_dir) if Path(f).suffix.lower() in DEFAULT_IMAGE_EXTS])
                print(f"Would resize {count} HR images in {hr_dir} to {hr_size}")
            else:
                print(f"HR directory not found: {hr_dir}")
        for lr_dir in lr_directories:
            if lr_dir.exists():
                count = len([f for f in os.listdir(lr_dir) if Path(f).suffix.lower() in DEFAULT_IMAGE_EXTS])
                print(f"Would resize {count} LR images in {lr_dir} to {lr_size}")
            else:
                print(f"LR directory not found: {lr_dir}")
        return

    print("=== RESIZING DATASET IMAGES ===")
    print(f"Target sizes: HR={hr_size}, LR={lr_size}")

    total_processed = 0

    # Process HR directories
    print(f"\n--- Processing HR Images ({hr_size[0]}x{hr_size[1]}) ---")
    for hr_dir in hr_directories:
        processed = resize_images_in_directory(hr_dir, hr_size, f"HR {hr_dir.parent.name}")
        total_processed += processed

    # Process LR directories
    print(f"\n--- Processing LR Images ({lr_size[0]}x{lr_size[1]}) ---")
    for lr_dir in lr_directories:
        processed = resize_images_in_directory(lr_dir, lr_size, f"LR {lr_dir.parent.name}")
        total_processed += processed

    print(f"\n=== SUMMARY ===")
    print(f"Total images resized: {total_processed}")
    print("Resize operation completed!")

if __name__ == "__main__":
    main()