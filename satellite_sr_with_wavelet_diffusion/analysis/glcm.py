import argparse
import glob
import os
from typing import Dict, List, Tuple

import cv2
import numpy as np

# Support both American and British spellings and newer module locations
try:
    from skimage.feature import graycomatrix, graycoprops  # type: ignore[attr-defined]
except Exception:
    try:
        from skimage.feature import greycomatrix as graycomatrix, greycoprops as graycoprops  # type: ignore[attr-defined]
    except Exception as e:  # pragma: no cover
        raise SystemExit("scikit-image is required (with texture features). Try: pip install --upgrade scikit-image") from e

try:
    import tqdm  # type: ignore
    def progress_iter(iterable, desc: str, unit: str):
        return tqdm.tqdm(iterable, desc=desc, unit=unit)
except Exception:
    tqdm = None
    def progress_iter(iterable, desc: str, unit: str):
        return iterable


PROPS = [
    "contrast",
    "dissimilarity",
    "homogeneity",
    "energy",
    "correlation",
    "ASM",
]


def read_gray_uint8(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to read image: {path}")
    # Ensure uint8 [0,255]
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return img


def compute_glcm_features(
    img: np.ndarray,
    *,
    distances: Tuple[int, ...] = (1, 2, 3),
    angles: Tuple[float, ...] = (0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4),
    levels: int = 256,
) -> Dict[str, float]:
    """Compute mean GLCM properties over distances and angles.

    Parameters:
    - img: uint8 grayscale image
    - distances: pixel distances
    - angles: angles in radians
    - levels: number of gray levels (uint8 -> 256)

    Returns:
    - dict of property -> mean value across all distances/angles
    """
    glcm = graycomatrix(
        img,
        distances=distances,
        angles=angles,
        levels=levels,
        symmetric=True,
        normed=True,
    )
    features: Dict[str, float] = {}
    for prop in PROPS:
        vals = graycoprops(glcm, prop)  # shape (len(distances), len(angles))
        features[prop] = float(np.mean(vals))
    return features


def find_images(folder: str, exts: Tuple[str, ...] = ("png", "jpg", "jpeg", "tif", "tiff")) -> List[str]:
    paths: List[str] = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(folder, f"*.{ext}")))
    return sorted(paths)


def write_csv(rows: List[Dict[str, str]], out_path: str) -> None:
    if not rows:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("")
        return
    headers = list(rows[0].keys())
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(",".join(headers) + "\n")
        for r in rows:
            f.write(",".join(str(r[h]) for h in headers) + "\n")


def process_folder(folder: str, label: str) -> List[Dict[str, str]]:
    paths = find_images(folder)
    rows: List[Dict[str, str]] = []
    for p in progress_iter(paths, desc=f"GLCM {label}", unit="img"):
        img = read_gray_uint8(p)
        feats = compute_glcm_features(img)
        row: Dict[str, str] = {"filename": os.path.basename(p), "set": label}
        for k, v in feats.items():
            row[k] = f"{v:.6f}"
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute GLCM features for images under root/{sr,hr}. Outputs CSVs."
    )
    parser.add_argument("root", type=str, help="Root folder containing 'sr' and 'hr' subfolders")
    parser.add_argument("--out", type=str, default=None, help="Output directory for CSVs (default: <root>)")
    parser.add_argument(
        "--distances", type=int, nargs="*", default=[1, 2, 3], help="GLCM distances"
    )
    parser.add_argument(
        "--angles", type=float, nargs="*", default=[0.0, 0.7853981634, 1.5707963268, 2.3561944902],
        help="GLCM angles in radians (default: 0, pi/4, pi/2, 3pi/4)"
    )
    args = parser.parse_args()

    sr_dir = os.path.join(args.root, "sr")
    hr_dir = os.path.join(args.root, "hr")
    if not os.path.isdir(sr_dir) or not os.path.isdir(hr_dir):
        raise SystemExit(f"Expected subfolders 'sr' and 'hr' under: {args.root}")

    out_dir = args.out or args.root
    os.makedirs(out_dir, exist_ok=True)

    # Process SR and HR
    sr_rows = process_folder(sr_dir, label="sr")
    hr_rows = process_folder(hr_dir, label="hr")

    # Write per-set CSVs
    sr_csv = os.path.join(out_dir, "glcm_sr.csv")
    hr_csv = os.path.join(out_dir, "glcm_hr.csv")
    write_csv(sr_rows, sr_csv)
    write_csv(hr_rows, hr_csv)

    # Optional combined CSV by filename if both sets present
    sr_map = {r["filename"]: r for r in sr_rows}
    combined: List[Dict[str, str]] = []
    for r in hr_rows:
        name = r["filename"]
        if name in sr_map:
            row: Dict[str, str] = {"filename": name}
            for prop in PROPS:
                row[f"sr_{prop}"] = sr_map[name][prop]
                row[f"hr_{prop}"] = r[prop]
            combined.append(row)
    if combined:
        write_csv(combined, os.path.join(out_dir, "glcm_sr_hr_pairs.csv"))

    print(f"Done. Wrote: {sr_csv}, {hr_csv}" + (", glcm_sr_hr_pairs.csv" if combined else ""))


if __name__ == "__main__":
    main()


