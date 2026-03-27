"""
UrbanSAR - SpaceNet-6 Data Download Script

Downloads SpaceNet-6 dataset from AWS S3 (official source).
Fallback: Kaggle community mirror.

Usage:
    python data/download_data.py                    # AWS S3 (default)
    python data/download_data.py --source kaggle    # Kaggle mirror
    python data/download_data.py --verify-only      # Just verify existing data
"""
import os
import sys
import subprocess
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RAW_DATA_DIR, SAR_DIR, OPTICAL_DIR, LABELS_DIR, S3_BUCKET


def download_from_s3(target_dir: Path, no_sign: bool = True) -> bool:
    """
    Download SpaceNet-6 from AWS S3.

    Args:
        target_dir: Local directory to download into
        no_sign: If True, use --no-sign-request (no AWS credentials needed).
                 If False, requires configured AWS credentials.

    Returns:
        True if download succeeded, False otherwise
    """
    cmd = [
        "aws", "s3", "cp",
        S3_BUCKET,
        str(target_dir),
        "--recursive"
    ]
    if no_sign:
        cmd.append("--no-sign-request")

    print(f"[INFO] Downloading SpaceNet-6 from AWS S3...")
    print(f"[INFO] Command: {' '.join(cmd)}")
    print(f"[INFO] Target: {target_dir}")
    print(f"[WARN] This is a large dataset (~30GB). Ensure sufficient disk space.")
    print()

    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        return True
    except subprocess.CalledProcessError as e:
        if no_sign:
            print(f"\n[WARN] --no-sign-request failed. SpaceNet-6 may require AWS credentials.")
            print(f"[INFO] To fix:")
            print(f"  1. Create a free AWS account at https://aws.amazon.com/")
            print(f"  2. Run: aws configure")
            print(f"  3. Re-run: python data/download_data.py")
            print(f"\n[INFO] Retrying with signed request...")
            return download_from_s3(target_dir, no_sign=False)
        else:
            print(f"[ERROR] AWS S3 download failed: {e}")
            return False
    except FileNotFoundError:
        print("[ERROR] AWS CLI not found. Install it: https://aws.amazon.com/cli/")
        print("[INFO] Or use: pip install awscli")
        return False


def download_from_kaggle(target_dir: Path) -> bool:
    """
    Download SpaceNet-6 from Kaggle (community mirror, may be incomplete).

    Requires:
        - Kaggle API key (~/.kaggle/kaggle.json)
        - pip install kaggle

    Returns:
        True if download succeeded, False otherwise
    """
    print("[INFO] Attempting Kaggle download (community mirror)...")
    print("[WARN] Kaggle mirror may not be complete or up-to-date.")

    try:
        import kaggle
        # Common SpaceNet-6 dataset slugs on Kaggle
        dataset_slug = "amerii/spacenet-6-multi-sensor-all-weather-mapping"
        cmd = [
            "kaggle", "datasets", "download",
            "-d", dataset_slug,
            "-p", str(target_dir),
            "--unzip"
        ]
        result = subprocess.run(cmd, check=True, capture_output=False)
        return True
    except Exception as e:
        print(f"[ERROR] Kaggle download failed: {e}")
        print("[INFO] Manual download options:")
        print(f"  1. Visit: https://www.kaggle.com/datasets/amerii/spacenet-6-multi-sensor-all-weather-mapping")
        print(f"  2. Download and extract to: {target_dir}")
        return False


def organize_data(raw_dir: Path) -> None:
    """
    Organize downloaded files into SAR/, optical/, labels/ subdirectories.
    SpaceNet-6 structure varies, so we detect and reorganize.
    """
    print("[INFO] Organizing data into SAR/optical/labels directories...")

    # Look for common SpaceNet-6 file patterns
    all_files = list(raw_dir.rglob("*.*"))

    sar_count = 0
    optical_count = 0
    label_count = 0

    for f in all_files:
        fname = f.name.lower()
        if f.is_file():
            # SAR images typically contain 'SAR' or 'sar' in name
            if "sar" in fname and fname.endswith((".tif", ".tiff")):
                dest = SAR_DIR / f.name
                if not dest.exists():
                    f.rename(dest)
                    sar_count += 1
            # Optical images
            elif ("optical" in fname or "ps-rgb" in fname or "ps_rgb" in fname) and fname.endswith((".tif", ".tiff")):
                dest = OPTICAL_DIR / f.name
                if not dest.exists():
                    f.rename(dest)
                    optical_count += 1
            # Label files (GeoJSON, CSV)
            elif fname.endswith((".geojson", ".json", ".csv")):
                dest = LABELS_DIR / f.name
                if not dest.exists():
                    f.rename(dest)
                    label_count += 1

    print(f"[INFO] Organized: {sar_count} SAR, {optical_count} optical, {label_count} label files")


def verify_data(raw_dir: Path) -> bool:
    """
    Verify downloaded data integrity.

    Returns:
        True if data looks valid, False otherwise
    """
    print("\n[INFO] Verifying data integrity...")

    sar_files = list(SAR_DIR.glob("*.tif")) + list(SAR_DIR.glob("*.tiff"))
    optical_files = list(OPTICAL_DIR.glob("*.tif")) + list(OPTICAL_DIR.glob("*.tiff"))
    label_files = list(LABELS_DIR.glob("*.geojson")) + list(LABELS_DIR.glob("*.json")) + list(LABELS_DIR.glob("*.csv"))

    print(f"  SAR images:     {len(sar_files)}")
    print(f"  Optical images: {len(optical_files)}")
    print(f"  Label files:    {len(label_files)}")

    if len(sar_files) == 0:
        print("[WARN] No SAR images found!")
        return False
    if len(optical_files) == 0:
        print("[WARN] No optical images found!")
        return False
    if len(label_files) == 0:
        print("[WARN] No label files found!")
        return False

    # Quick sanity check: try reading first SAR file
    try:
        import rasterio
        with rasterio.open(sar_files[0]) as src:
            bands = src.count
            width = src.width
            height = src.height
            print(f"\n  Sample SAR: {sar_files[0].name}")
            print(f"    Bands: {bands}, Size: {width}x{height}")
            if bands not in [1, 4]:
                print(f"  [WARN] Expected 1 or 4 bands for SAR, got {bands}")
    except Exception as e:
        print(f"  [WARN] Could not read SAR file for verification: {e}")

    print("\n[OK] Data verification complete.")
    return True


def main():
    parser = argparse.ArgumentParser(description="Download SpaceNet-6 dataset")
    parser.add_argument("--source", choices=["s3", "kaggle"], default="s3",
                        help="Download source (default: s3)")
    parser.add_argument("--verify-only", action="store_true",
                        help="Only verify existing data, don't download")
    args = parser.parse_args()

    if args.verify_only:
        verify_data(RAW_DATA_DIR)
        return

    print("=" * 60)
    print("UrbanSAR - SpaceNet-6 Data Download")
    print("=" * 60)

    success = False

    if args.source == "s3":
        success = download_from_s3(RAW_DATA_DIR)
        if not success:
            print("\n[INFO] Falling back to Kaggle...")
            success = download_from_kaggle(RAW_DATA_DIR)
    else:
        success = download_from_kaggle(RAW_DATA_DIR)

    if success:
        organize_data(RAW_DATA_DIR)
        verify_data(RAW_DATA_DIR)
        print("\n[OK] Download complete!")
    else:
        print("\n[ERROR] All download methods failed.")
        print("[INFO] Manual download instructions:")
        print(f"  1. AWS: aws s3 cp {S3_BUCKET} {RAW_DATA_DIR} --recursive")
        print(f"  2. Kaggle: https://www.kaggle.com/datasets/amerii/spacenet-6-multi-sensor-all-weather-mapping")
        print(f"  3. Extract data into: {RAW_DATA_DIR}")
        sys.exit(1)


if __name__ == "__main__":
    main()
