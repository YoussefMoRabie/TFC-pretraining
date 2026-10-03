#!/usr/bin/env python3
"""
download_datasets.py

Robust dataset downloader for TFC-pretraining repository.
Fetches dataset files directly via Figshare API v2 endpoint:
https://api.figshare.com/v2/articles/<ARTICLE_ID>/versions/<VERSION>/files

Supports macOS and Linux, handles SSL contexts, stream downloads with progress,
validates non-zero byte sizes, and extracts ZIP archives if encountered.
"""

import os
import sys
import json
import ssl
import zipfile
import argparse
from pathlib import Path
import urllib.request
import urllib.error

# Figshare Article / Version Mapping for TFC Datasets
DATASETS = [
    {"name": "SleepEEG", "article_id": 19930178, "version": 1},
    {"name": "Epilepsy", "article_id": 19930199, "version": 2},
    {"name": "FD-A",     "article_id": 19930205, "version": 1},
    {"name": "FD-B",     "article_id": 19930226, "version": 1},
    {"name": "HAR",      "article_id": 19930244, "version": 1},
    {"name": "Gesture",  "article_id": 19930247, "version": 1},
    {"name": "ECG",      "article_id": 19930253, "version": 1},
    {"name": "EMG",      "article_id": 19930250, "version": 1},
]

REQUIRED_FILES = ["train.pt", "val.pt", "test.pt"]


def get_ssl_context():
    """Create SSL context with fallback for systems with certificate verification issues."""
    try:
        ctx = ssl.create_default_context()
        return ctx
    except Exception:
        return ssl._create_unverified_context()


def is_dataset_complete(target_dir: Path) -> bool:
    """Check if train.pt, val.pt, and test.pt exist and have non-zero size."""
    for req_file in REQUIRED_FILES:
        file_path = target_dir / req_file
        if not file_path.exists() or file_path.stat().st_size == 0:
            return False
    return True


def query_figshare_files(article_id: int, version: int, ssl_ctx: ssl.SSLContext) -> list:
    """Query Figshare API v2 for file metadata of an article version."""
    url = f"https://api.figshare.com/v2/articles/{article_id}/versions/{version}/files"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})

    try:
        with urllib.request.urlopen(req, context=ssl_ctx, timeout=30) as resp:
            if resp.status != 200:
                raise RuntimeError(f"HTTP Error {resp.status} fetching API metadata from {url}")
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        # Fallback to unverified SSL context if standard context fails (common on macOS)
        unverified_ctx = ssl._create_unverified_context()
        with urllib.request.urlopen(req, context=unverified_ctx, timeout=30) as resp:
            if resp.status != 200:
                raise RuntimeError(f"HTTP Error {resp.status} fetching API metadata from {url}")
            return json.loads(resp.read().decode("utf-8"))


def download_file(url: str, dest_path: Path, expected_size: int, ssl_ctx: ssl.SSLContext):
    """Download file from url to dest_path with progress display and validation."""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})

    def _do_download(ctx):
        with urllib.request.urlopen(req, context=ctx, timeout=120) as resp:
            if resp.status != 200:
                raise RuntimeError(f"Download failed with HTTP status {resp.status} for {url}")

            total_size = int(resp.headers.get("Content-Length", expected_size or 0))
            downloaded = 0
            chunk_size = 1024 * 1024  # 1MB chunks

            tmp_path = dest_path.with_suffix(dest_path.suffix + ".tmp")
            with open(tmp_path, "wb") as f:
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        pct = int(downloaded * 100 / total_size)
                        mb_done = downloaded / (1024 * 1024)
                        mb_total = total_size / (1024 * 1024)
                        sys.stdout.write(f"\r  Downloading {dest_path.name} ({mb_done:.1f}/{mb_total:.1f} MB - {pct}%)...")
                        sys.stdout.flush()
                    else:
                        mb_done = downloaded / (1024 * 1024)
                        sys.stdout.write(f"\r  Downloading {dest_path.name} ({mb_done:.1f} MB)...")
                        sys.stdout.flush()
            sys.stdout.write("\n")

            if downloaded == 0:
                if tmp_path.exists():
                    tmp_path.unlink()
                raise RuntimeError(f"Downloaded file {dest_path.name} is 0 bytes!")

            if expected_size > 0 and downloaded != expected_size:
                sys.stdout.write(f"  [Warning] Size mismatch for {dest_path.name}: expected {expected_size}, got {downloaded}\n")

            tmp_path.replace(dest_path)

    try:
        _do_download(ssl_ctx)
    except urllib.error.URLError:
        unverified_ctx = ssl._create_unverified_context()
        _do_download(unverified_ctx)


def process_dataset(idx: int, total: int, ds_info: dict, base_dir: Path, force: bool, ssl_ctx: ssl.SSLContext) -> bool:
    name = ds_info["name"]
    art_id = ds_info["article_id"]
    ver = ds_info["version"]

    target_dir = base_dir / "datasets" / name
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{idx}/{total}] Downloading {name}...")

    if not force and is_dataset_complete(target_dir):
        print(f"  Dataset {name} already complete in {target_dir}. Skipping.")
        print(f"{name} complete.\n")
        return True

    try:
        files_info = query_figshare_files(art_id, ver, ssl_ctx)
        print(f"Found {len(files_info)} file(s)")

        for file_item in files_info:
            fname = file_item["name"]
            fsize = file_item.get("size", 0)
            durl = file_item["download_url"]

            dest_path = target_dir / fname

            if not force and dest_path.exists() and dest_path.stat().st_size == fsize and fsize > 0:
                print(f"  {fname} already downloaded ({fsize / (1024*1024):.2f} MB). Skipping.")
                continue

            download_file(durl, dest_path, fsize, ssl_ctx)

            # If downloaded file is a ZIP archive, test integrity and extract
            if fname.lower().endswith(".zip") and zipfile.is_zipfile(dest_path):
                print(f"  Extracting {fname} to {target_dir}...")
                with zipfile.ZipFile(dest_path, "r") as z:
                    bad_file = z.testzip()
                    if bad_file:
                        raise RuntimeError(f"Corrupted ZIP archive {fname}: bad file {bad_file}")
                    z.extractall(target_dir)
                dest_path.unlink()
                print(f"  Extracted and removed {fname}.")

        if is_dataset_complete(target_dir):
            print(f"{name} complete.\n")
            return True
        else:
            print(f"[Warning] {name} download finished, but required files check failed.\n")
            return False
    except Exception as e:
        print(f"Error downloading dataset {name}: {e}\n")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download TF-C datasets from Figshare API v2.")
    parser.add_argument("--force", action="store_true", help="Force re-download even if dataset files exist.")
    parser.add_argument("--dataset", type=str, default=None, help="Download a specific dataset by name.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    ssl_ctx = get_ssl_context()

    target_datasets = DATASETS
    if args.dataset:
        target_datasets = [d for d in DATASETS if d["name"].lower() == args.dataset.lower()]
        if not target_datasets:
            print(f"Error: Unknown dataset '{args.dataset}'. Available: {[d['name'] for d in DATASETS]}")
            sys.exit(1)

    print("==================================================")
    print("      TF-C Dataset Downloader (Figshare API)      ")
    print("==================================================\n")

    results = {}
    for idx, ds_info in enumerate(target_datasets, 1):
        success = process_dataset(idx, len(target_datasets), ds_info, repo_root, args.force, ssl_ctx)
        results[ds_info["name"]] = success

    print("==================================================")
    print("                Download Summary                  ")
    print("==================================================")
    all_passed = True
    for name, ok in results.items():
        status = "SUCCESS" if ok else "FAILED"
        if not ok:
            all_passed = False
        print(f"  {name:10s} : {status}")

    if not all_passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
