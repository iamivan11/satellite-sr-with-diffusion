"""
Match the color distribution of LR images to their corresponding HR images
for a generic dataset.

By default, writes color-matched results to sibling directories named
`lr_matched` next to each `lr` directory to avoid destructive edits.
Use `--inplace` to overwrite the original LR images.

Supports two methods:
- hist: histogram matching per channel (requires scikit-image)
- meanstd: simple per-channel mean/std matching (no extra deps)

Folder structure expected (per split):
- <base>/train/hr
- <base>/train/lr
- <base>/test/hr
- <base>/test/lr

Filenames in `hr` and `lr` are assumed to correspond 1:1.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Iterable
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def try_import_match_histograms():
    """Lazily try to import skimage's match_histograms, returning a callable or None."""
    try:
        from skimage.exposure import match_histograms as _match_histograms

        # Wrap to unify the API across skimage versions
        def match_fn(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
            try:
                # Newer API uses channel_axis
                return _match_histograms(source, reference, channel_axis=-1)
            except TypeError:
                # Older API uses multichannel
                return _match_histograms(source, reference, multichannel=True)

        return match_fn
    except Exception:
        return None


DEFAULT_BASE_DIR: Path = Path("data/DATASET")
DEFAULT_SPLITS: list[str] = ["train", "test", "val"]
DEFAULT_METHOD: str = "hist"
DEFAULT_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def list_image_files(directory: Path) -> list[str]:
    try:
        return sorted([f for f in os.listdir(directory) if Path(f).suffix.lower() in DEFAULT_EXTS])
    except FileNotFoundError:
        return []


def mean_std_color_match(
    source_bgr: np.ndarray, reference_bgr: np.ndarray, eps: float = 1e-6
) -> np.ndarray:
    """Match source image colors to reference using per-channel mean/std normalization.

    Works for arbitrary sizes; channels are assumed BGR.
    """
    source = source_bgr.astype(np.float32)
    reference = reference_bgr.astype(np.float32)

    # Compute stats on each channel independently
    src_means = source.reshape(-1, 3).mean(axis=0)
    ref_means = reference.reshape(-1, 3).mean(axis=0)
    src_stds = source.reshape(-1, 3).std(axis=0)
    ref_stds = reference.reshape(-1, 3).std(axis=0)

    # Avoid division by zero
    src_stds = np.maximum(src_stds, eps)

    matched = (source - src_means) * (ref_stds / src_stds) + ref_means
    return np.clip(matched, 0, 255).astype(np.uint8)


def color_match_pair(
    lr_img_bgr: np.ndarray,
    hr_img_bgr: np.ndarray,
    method: str,
    skimage_match_histograms,
) -> np.ndarray:
    if method == "hist":
        if skimage_match_histograms is None:
            raise RuntimeError(
                "Histogram matching requires scikit-image. Install it or use --method meanstd."
            )
        # skimage expects contiguous arrays
        src = np.ascontiguousarray(lr_img_bgr)
        ref = np.ascontiguousarray(hr_img_bgr)
        matched = skimage_match_histograms(src, ref)
        return np.clip(matched, 0, 255).astype(np.uint8)
    if method == "meanstd":
        return mean_std_color_match(lr_img_bgr, hr_img_bgr)
    raise ValueError(f"Unknown method: {method}")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def process_split(
    base_dir: Path,
    split: str,
    method: str,
    inplace: bool,
    dry_run: bool,
) -> tuple[int, int, int]:
    """Process one split (e.g., train or test).

    Returns (num_processed, num_missing_pairs, num_errors)
    """
    hr_dir = base_dir / split / "hr"
    lr_dir = base_dir / split / "lr"
    out_dir = lr_dir if inplace else (base_dir / split / "lr_matched")

    hr_files = set(list_image_files(hr_dir))
    lr_files = set(list_image_files(lr_dir))

    if not hr_files:
        print(f"[WARN] No HR images found in {hr_dir}")
    if not lr_files:
        print(f"[WARN] No LR images found in {lr_dir}")

    common = sorted(hr_files & lr_files)
    missing_pairs = len(hr_files | lr_files) - len(common)

    if dry_run:
        target_note = "(in-place)" if inplace else f"-> {out_dir}"
        print(
            f"Split '{split}': would color-match {len(common)} pairs {target_note}; "
            f"{missing_pairs} unmatched filenames"
        )
        return (0, missing_pairs, 0)

    if not inplace:
        ensure_dir(out_dir)

    match_hist_fn = try_import_match_histograms() if method == "hist" else None

    processed = 0
    errors = 0

    desc = f"{split} ({'hist' if method == 'hist' else 'meanstd'})"
    for filename in tqdm(common, desc=desc):
        try:
            lr_path = lr_dir / filename
            hr_path = hr_dir / filename

            lr_img = cv2.imread(str(lr_path), cv2.IMREAD_COLOR)
            hr_img = cv2.imread(str(hr_path), cv2.IMREAD_COLOR)

            if lr_img is None or hr_img is None:
                errors += 1
                continue

            matched = color_match_pair(lr_img, hr_img, method, match_hist_fn)

            # Preserve original spatial resolution of LR image
            save_path = out_dir / filename
            ok = cv2.imwrite(str(save_path), matched)
            if not ok:
                errors += 1
                continue

            processed += 1
        except Exception as e:
            errors += 1
            print(f"Error processing {filename}: {e}")

    return (processed, missing_pairs, errors)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Match LR image colors to HR references in OLI2MSI. "
            "Defaults to non-destructive output in 'lr_matched'."
        )
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("data/OLI2MSI"),
        help="Base directory containing splits (train/test[/val])",
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="*",
        default=DEFAULT_SPLITS,
        help="Splits to process (default: train test)",
    )
    parser.add_argument(
        "--method",
        choices=["hist", "meanstd"],
        default=DEFAULT_METHOD,
        help="Color matching method: 'hist' (requires scikit-image) or 'meanstd'",
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Overwrite LR images in-place (destructive)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List actions without writing any files",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)

    if args.inplace and not args.dry_run:
        print("--- WARNING ---")
        print("This will OVERWRITE images in LR directories.")
        print("There is no undo. Consider running without --inplace first.")
        response = input("Continue? (y/n): ").strip().lower()
        if response != "y":
            print("Operation cancelled.")
            return

    print("=== COLOR MATCH LR -> HR (OLI2MSI) ===")
    print(f"Base dir: {args.base_dir}")
    print(f"Splits: {', '.join(args.splits)}")
    print(f"Method: {args.method}")
    print(f"Target: {'in-place (lr)' if args.inplace else 'lr_matched output'}")

    totals = {
        "processed": 0,
        "missing": 0,
        "errors": 0,
    }

    for split in args.splits:
        processed, missing, errors = process_split(
            base_dir=args.base_dir,
            split=split,
            method=args.method,
            inplace=args.inplace,
            dry_run=args.dry_run,
        )
        totals["processed"] += processed
        totals["missing"] += missing
        totals["errors"] += errors

    print("\n=== SUMMARY ===")
    if args.dry_run:
        print("Dry run completed.")
    print(f"Pairs processed: {totals['processed']}")
    print(f"Unmatched filenames: {totals['missing']}")
    print(f"Errors: {totals['errors']}")
    if not args.dry_run:
        print("Color matching completed.")


if __name__ == "__main__":
    main()
