import argparse
import glob
import os

import cv2
import numpy as np

try:
    from skimage.feature import local_binary_pattern
except Exception as e:  # pragma: no cover
    raise SystemExit("scikit-image is required. Install with: pip install scikit-image") from e

try:
    import tqdm  # type: ignore

    def progress_iter(iterable, desc: str, unit: str):
        return tqdm.tqdm(iterable, desc=desc, unit=unit)
except Exception:
    tqdm = None

    def progress_iter(iterable, desc: str, unit: str):
        return iterable


def read_gray_uint8(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to read image: {path}")
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return img


def compute_lbp_hist(
    img: np.ndarray,
    *,
    points: int = 8,
    radius: float = 1.0,
    method: str = "uniform",
) -> dict[str, float]:
    """Compute normalized histogram of LBP codes.

    Parameters:
    - img: uint8 grayscale image
    - points: number of circularly symmetric neighbor set points
    - radius: radius of circle
    - method: LBP method (e.g., 'uniform')

    Returns:
    - dict mapping 'bin_0'...'bin_{n_bins-1}' -> probability
    """
    lbp = local_binary_pattern(img, P=points, R=radius, method=method)

    # Number of bins: for 'uniform' with P points it's P + 2; otherwise cover full range
    n_bins = points + 2 if method == "uniform" else int(lbp.max() + 1)

    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_bins + 1), range=(0, n_bins))
    hist = hist.astype(np.float64)
    hist_sum = hist.sum()
    if hist_sum > 0:
        hist /= hist_sum

    features: dict[str, float] = {f"bin_{i}": float(hist[i]) for i in range(n_bins)}
    return features


def find_images(
    folder: str, exts: tuple[str, ...] = ("png", "jpg", "jpeg", "tif", "tiff")
) -> list[str]:
    paths: list[str] = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(folder, f"*.{ext}")))
    return sorted(paths)


def write_csv(rows: list[dict[str, str]], out_path: str) -> None:
    if not rows:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("")
        return
    headers = list(rows[0].keys())
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(",".join(headers) + "\n")
        for r in rows:
            f.write(",".join(str(r[h]) for h in headers) + "\n")


def process_folder(
    folder: str, label: str, points: int, radius: float, method: str
) -> list[dict[str, str]]:
    paths = find_images(folder)
    rows: list[dict[str, str]] = []
    for p in progress_iter(paths, desc=f"LBP {label}", unit="img"):
        img = read_gray_uint8(p)
        feats = compute_lbp_hist(img, points=points, radius=radius, method=method)
        row: dict[str, str] = {"filename": os.path.basename(p), "set": label}
        for k, v in feats.items():
            row[k] = f"{v:.6f}"
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute LBP histograms for images under root/{sr,hr}. Outputs CSVs."
    )
    parser.add_argument("root", type=str, help="Root folder containing 'sr' and 'hr' subfolders")
    parser.add_argument(
        "--out", type=str, default=None, help="Output directory for CSVs (default: <root>)"
    )
    parser.add_argument("--points", type=int, default=8, help="LBP points (P)")
    parser.add_argument("--radius", type=float, default=1.0, help="LBP radius (R)")
    parser.add_argument("--method", type=str, default="uniform", help="LBP method (e.g., uniform)")
    args = parser.parse_args()

    sr_dir = os.path.join(args.root, "sr")
    hr_dir = os.path.join(args.root, "hr")
    if not os.path.isdir(sr_dir) or not os.path.isdir(hr_dir):
        raise SystemExit(f"Expected subfolders 'sr' and 'hr' under: {args.root}")

    out_dir = args.out or args.root
    os.makedirs(out_dir, exist_ok=True)

    # Process SR and HR
    sr_rows = process_folder(
        sr_dir, label="sr", points=args.points, radius=args.radius, method=args.method
    )
    hr_rows = process_folder(
        hr_dir, label="hr", points=args.points, radius=args.radius, method=args.method
    )

    # Write per-set CSVs
    sr_csv = os.path.join(out_dir, "lbp_sr.csv")
    hr_csv = os.path.join(out_dir, "lbp_hr.csv")
    write_csv(sr_rows, sr_csv)
    write_csv(hr_rows, hr_csv)

    # Optional combined CSV by filename if both sets present (matching bins)
    # Determine common bin keys from SR rows if any
    bin_keys: list[str] = [k for k in sr_rows[0] if k.startswith("bin_")] if sr_rows else []
    sr_map = {r["filename"]: r for r in sr_rows}
    combined: list[dict[str, str]] = []
    for r in hr_rows:
        name = r["filename"]
        if name in sr_map:
            row: dict[str, str] = {"filename": name}
            for bk in bin_keys:
                row[f"sr_{bk}"] = sr_map[name][bk]
                row[f"hr_{bk}"] = r[bk]
            combined.append(row)
    if combined:
        write_csv(combined, os.path.join(out_dir, "lbp_sr_hr_pairs.csv"))

    print(f"Done. Wrote: {sr_csv}, {hr_csv}" + (", lbp_sr_hr_pairs.csv" if combined else ""))


if __name__ == "__main__":
    main()
