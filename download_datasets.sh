#!/usr/bin/env bash
# TFC Dataset Downloader Wrapper
# Executes download_datasets.py using Python 3

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

python3 download_datasets.py "$@"
