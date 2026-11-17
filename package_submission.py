#!/usr/bin/env python
"""
Helper script to package clean source-only zip archive for Google Form submission.
Omits datasets, virtual environments, trained models, MLflow database runs, and caches.
"""

import os
import zipfile
from pathlib import Path


def create_submission_zip(output_zip: str = "bangla-ocr-submission.zip"):
    root = Path(__file__).resolve().parent

    # Explicit list of allowed items or exclusion filters
    excluded_names = {
        "venv", ".venv", "env", "ENV",
        "__pycache__", ".pytest_cache", ".git", ".harness",
        "Datasets", "data", "models", "artifacts",
        output_zip
    }

    print(f"Creating clean source code submission zip: {output_zip}...")
    file_count = 0

    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in root.rglob("*"):
            if not p.is_file():
                continue
            # Check if any parent part is in excluded
            parts = p.relative_to(root).parts
            if any(part in excluded_names for part in parts):
                continue
            # Also exclude any .zip or .pyc
            if p.suffix in [".zip", ".pyc", ".log", ".db"]:
                continue

            arcname = p.relative_to(root).as_posix()
            zf.write(p, arcname)
            file_count += 1
            print(f"  + Added: {arcname}")

    zip_size = os.path.getsize(output_zip) / (1024 * 1024)
    print(f"\n[DONE] Package created successfully: {output_zip} ({file_count} files, {zip_size:.2f} MB)")


if __name__ == "__main__":
    create_submission_zip()
