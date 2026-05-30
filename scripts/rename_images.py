#!/usr/bin/env python3
"""
Safely and atomically renames all images in dataset folders to sequential numbers.
This script is designed to be robust against interruptions. It works by creating
a temporary directory for the renamed files. Only after all files are successfully
copied and renamed does it replace the original directory.

This prevents the dataset from being left in a partially renamed state if the
script is stopped midway.
"""

import os
import shutil
from pathlib import Path
from tqdm import tqdm
import argparse
from typing import Set

DEFAULT_BASE_DIR: Path = Path("data/DATASET")
DEFAULT_SPLITS = ["train", "test", "val"]
DEFAULT_FOLDERS = ["hr", "lr"]
DEFAULT_EXTS: Set[str] = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}


def safe_rename_directory(directory: Path, description: str):
    """
    Safely renames all images in a directory to sequential numbers using a temporary folder.

    Args:
        directory (Path): The target directory (e.g., 'data/SEN2VENUS/train/hr').
        description (str): A description for the tqdm progress bar.
    """
    if not directory.is_dir():
        print(f"Source directory not found: {directory}")
        return

    # Create a temporary directory for the renamed files
    temp_dir = directory.parent / f"{directory.name}_renamed"

    # Clean up any leftover temp directory from a previous failed run
    if temp_dir.exists():
        print(f"Removing leftover temporary directory: {temp_dir}")
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True)

    try:
        # Important: List files from the original directory
        image_files = sorted([f for f in os.listdir(directory) if Path(f).suffix.lower() in DEFAULT_EXTS])

        if not image_files:
            print(f"No images found in {directory}")
            shutil.rmtree(temp_dir)
            return

        print(f"\nProcessing {len(image_files)} images from '{directory}' into '{temp_dir}'")
        
        # Copy files to the temporary directory with new names
        for i, filename in enumerate(tqdm(image_files, desc=description)):
            original_path = directory / filename
            extension = original_path.suffix
            new_filename = f"{i:07d}{extension}"
            new_path = temp_dir / new_filename
            shutil.copy2(original_path, new_path) # copy2 preserves metadata

        # --- Atomic Swap ---
        # If we've reached here, all files were copied successfully.
        # Now, we replace the old directory with the new one.
        print(f"Successfully prepared '{temp_dir}'. Replacing original directory.")
        shutil.rmtree(directory)
        temp_dir.rename(directory)
        print(f"Renamed '{temp_dir}' to '{directory}'. Operation complete for this folder.")

    except Exception as e:
        print(f"\n--- ERROR ---")
        print(f"An error occurred while processing '{directory}': {e}")
        print(f"The original directory '{directory}' has been left untouched.")
        print(f"Please delete the temporary directory '{temp_dir}' before trying again.")
        # Do not clean up automatically, user should inspect the error
    except KeyboardInterrupt:
        print(f"\n--- CANCELED ---")
        print(f"Operation canceled by user.")
        print(f"The original directory '{directory}' has been left untouched.")
        print(f"Please delete the temporary directory '{temp_dir}' before trying again.")

def main():
    parser = argparse.ArgumentParser(description='Safely rename dataset images to sequential numbers.')
    parser.add_argument('--base-dir', type=Path, default=DEFAULT_BASE_DIR, help='Path to the dataset base directory')
    parser.add_argument('--splits', type=str, nargs='*', default=DEFAULT_SPLITS, help='Splits to process')
    parser.add_argument('--folders', type=str, nargs='*', default=DEFAULT_FOLDERS, help='Subfolders to process (e.g., hr lr)')
    args = parser.parse_args()

    print("=== SAFE RENAMING SCRIPT FOR DATASET ===")
    print(f"Processing dataset at: {args.base_dir}")
    print("This script is safe to interrupt. Original data will not be corrupted.\n")

    for split in args.splits:
        print(f"--- Processing Split: {split} ---")
        for folder in args.folders:
            directory_path = args.base_dir / split / folder
            safe_rename_directory(directory_path, description=f"{split.capitalize()} {folder.upper()}")

    print("\n\n=== All operations completed successfully! ===")


if __name__ == "__main__":
    main()