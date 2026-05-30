"""
Upsamples all images in the dataset low-resolution (lr) directories
to a target resolution (default 192x192) using bicubic interpolation.

This script directly overwrites the original files.
"""
import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import argparse
from typing import Set


DEFAULT_BASE_DIR: Path = Path("data/DATASET")
DEFAULT_TARGET_EDGE: int = 192
DEFAULT_EXTS: Set[str] = {'.jpg', '.jpeg', '.png'}


def upsample_images_in_directory(directory: Path, target_size: tuple):
    """
    Upsamples all images in a directory to the target size.

    Args:
        directory (Path): The directory containing the images to upsample.
        target_size (tuple): The target resolution, e.g., (192, 192).
    """
    if not directory.is_dir():
        print(f"Directory not found: {directory}, skipping.")
        return

    image_files = sorted([f for f in os.listdir(directory) if Path(f).suffix.lower() in DEFAULT_EXTS])

    if not image_files:
        print(f"No images found in {directory}.")
        return

    print(f"\nUpsampling {len(image_files)} images in '{directory}' to {target_size[0]}x{target_size[1]}...")

    for filename in tqdm(image_files, desc=f"Upsampling {directory.parent.name}/{directory.name}"):
        try:
            image_path = directory / filename
            with Image.open(image_path) as img:
                # Ensure image is in a standard mode like RGB before saving as JPEG
                if img.mode not in ['RGB']:
                    img = img.convert('RGB')
                
                # Upsample using bicubic interpolation
                upsampled_img = img.resize(target_size, Image.BICUBIC)
                
                # Overwrite the original file
                upsampled_img.save(image_path, 'JPEG', quality=100)

        except Exception as e:
            print(f"Error processing {filename}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Bicubically upsample LR images in a dataset.',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--base-dir', 
        type=Path, 
        default=DEFAULT_BASE_DIR,
        help='Path to the dataset base directory.'
    )
    parser.add_argument(
        '--target-size',
        type=int,
        default=DEFAULT_TARGET_EDGE,
        help='The target edge size for the upsampled images (e.g., 192 for 192x192).'
    )
    
    print("--- WARNING ---")
    print("This will OVERWRITE low-resolution (lr) images.")
    print("There is no undo. Consider backing up your data.")

    args = parser.parse_args()
    
    size = (args.target_size, args.target_size)
    
    print("\n=== UPSAMPLING SEN2VENUS LR IMAGES ===")
    
    # Define all LR directories to process
    lr_directories = [
        args.base_dir / "train" / "lr",
        args.base_dir / "test" / "lr",
        args.base_dir / "val" / "lr",
    ]

    for directory in lr_directories:
        upsample_images_in_directory(directory, size)

    print("\n=== All upsampling operations completed successfully! ===")

if __name__ == "__main__":
    main()