#!/usr/bin/env python
"""
Helper script to package clean source-only zip archive for Google Form submission.
Omits datasets, virtual environments, trained models, MLflow database runs, and caches.
Complies with Assignment 04 Section 2 requirements.
"""

import os
import zipfile
from pathlib import Path


def create_submission_zip(output_zip: str = "bangla-ocr-submission.zip"):
    root = Path(__file__).resolve().parent

    # Explicit list of excluded names (case-insensitive checking)
    excluded_names = {
        "venv", ".venv", "env", "ENV",
        "__pycache__", ".pytest_cache", ".git", ".github", ".harness",
        "datasets", "Datasets", "data", "Data",
        "models", "artifacts",
        output_zip
    }

    excluded_extensions = {".zip", ".pyc", ".log", ".db", ".keras", ".h5", ".tar", ".gz", ".whl"}

    print(f"Creating clean source code submission zip: {output_zip}...")
    file_count = 0

    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in root.rglob("*"):
            if not p.is_file():
                continue
            # Check if any parent part is in excluded
            parts = p.relative_to(root).parts
            if any(part in excluded_names or part.lower() in [e.lower() for e in excluded_names] for part in parts):
                continue
            # Exclude large binaries/artifacts
            if p.suffix.lower() in excluded_extensions:
                continue

            arcname = p.relative_to(root).as_posix()
            zf.write(p, arcname)
            file_count += 1
            print(f"  + Added: {arcname}")

    zip_size = os.path.getsize(output_zip) / (1024 * 1024)
    print(f"\n[DONE] Package created successfully: {output_zip} ({file_count} files, {zip_size:.2f} MB)")


if __name__ == "__main__":
    create_submission_zip()
