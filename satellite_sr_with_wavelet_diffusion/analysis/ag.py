import argparse
import glob
import os
from typing import List

import cv2
import numpy as np

try:
    import tqdm  # type: ignore
    def progress_iter(iterable, desc: str, unit: str):
        return tqdm.tqdm(iterable, desc=desc, unit=unit)
except Exception:
    tqdm = None
    def progress_iter(iterable, desc: str, unit: str):
        return iterable


def compute_average_gradient(image_path: str) -> float:
    """Compute Average Gradient (AG) for a single image.

    Parameters:
    - image_path: path to the image file

    Returns:
    - Mean gradient magnitude across all pixels (float)
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to read image: {image_path}")

    # Sobel gradients in X and Y (float for precision)
    grad_x = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)

    # Gradient magnitude per pixel
    magnitude = cv2.magnitude(grad_x, grad_y)

    # Average gradient (AG)
    return float(np.mean(magnitude))


def find_sr_images(folder: str, ext: str) -> List[str]:
    """Find all super-res images matching *_sr.<ext> in a folder."""
    pattern = os.path.join(folder, f"*_sr.{ext}")
    return sorted(glob.glob(pattern))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute average 'Average Gradient (AG)' over *_sr images in a folder.")
    parser.add_argument("folder", type=str, help="Path to folder containing images (e.g., /path/to/dir)")
    parser.add_argument("--ext", type=str, default="png", help="Image extension to match (default: png)")
    parser.add_argument("--per-image", action="store_true", help="Also print per-image AG values")
    args = parser.parse_args()

    sr_images = find_sr_images(args.folder, args.ext)
    if not sr_images:
        raise SystemExit(f"No images found matching '*_sr.{args.ext}' in: {args.folder}")

    ag_values: List[float] = []
    print(f"Found {len(sr_images)} images matching '*_sr.{args.ext}'.")
    for path in progress_iter(sr_images, desc="Computing AG", unit="img"):
        ag = compute_average_gradient(path)
        ag_values.append(ag)
        if args.per_image:
            print(f"{os.path.basename(path)}\tAG={ag:.6f}")

    dataset_avg_ag = float(np.mean(ag_values))
    print(f"Count={len(sr_images)}\tAverage_AG={dataset_avg_ag:.6f}")


if __name__ == "__main__":
    main()


