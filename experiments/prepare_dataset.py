#!/usr/bin/env python3
"""
Step 1: Dataset Preparation
============================
Downloads book.txt (A Christmas Carol) and splits it into deterministic halves.

Usage:
    python experiments/prepare_dataset.py

Outputs:
    data/splits/half_a.txt   — first half of paragraphs
    data/splits/half_b.txt   — second half of paragraphs
    data/splits/full.txt     — complete text
"""

import os
import sys
import hashlib
import urllib.request
import urllib.error
import time

# Resolve project root (one level up from experiments/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

BOOK_URLS = [
    "https://raw.githubusercontent.com/HKUDS/LightRAG/main/examples/book.txt",
    "https://www.gutenberg.org/cache/epub/46/pg46.txt",  # A Christmas Carol fallback
]
BOOK_URL = BOOK_URLS[0]  # Primary (kept for compatibility)
BOOK_PATH = os.path.join(PROJECT_DIR, "data", "book.txt")
SPLITS_DIR = os.path.join(PROJECT_DIR, "data", "splits")

MAX_RETRIES = 3
RETRY_DELAY = 5


def download_book(url: str, dest: str, retries: int = MAX_RETRIES) -> None:
    """Download book.txt with retry logic."""
    if os.path.exists(dest):
        size = os.path.getsize(dest)
        if size > 10000:  # Sanity check: file should be at least 10KB
            print(f"  ✓ book.txt already exists ({size:,} bytes), skipping download")
            return
        else:
            print(f"  ⚠ book.txt exists but is only {size} bytes — re-downloading")

    os.makedirs(os.path.dirname(dest), exist_ok=True)

    for attempt in range(1, retries + 1):
        try:
            print(f"  Downloading from {url} (attempt {attempt}/{retries})...")
            urllib.request.urlretrieve(url, dest)
            size = os.path.getsize(dest)
            print(f"  ✓ Downloaded {size:,} bytes")

            # Validate it's actual text content
            with open(dest, "r", encoding="utf-8") as f:
                first_line = f.readline().strip()
            if len(first_line) < 5:
                raise ValueError(f"Downloaded file appears empty (first line: '{first_line}')")

            return
        except (urllib.error.URLError, urllib.error.HTTPError, ValueError) as e:
            print(f"  ✗ Attempt {attempt} failed: {e}")
            if attempt < retries:
                print(f"    Retrying in {RETRY_DELAY}s...")
                time.sleep(RETRY_DELAY)
            else:
                print(f"\n  FATAL: Could not download book.txt after {retries} attempts.")
                print(f"  Please download manually from:")
                print(f"    {url}")
                print(f"  And place it at:")
                print(f"    {dest}")
                sys.exit(1)


def split_corpus(input_path: str, output_dir: str) -> dict:
    """Split corpus into two halves by paragraph boundary."""
    os.makedirs(output_dir, exist_ok=True)

    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Split by double newlines (paragraph boundaries)
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    if len(paragraphs) < 4:
        print(f"  ⚠ Only {len(paragraphs)} paragraphs found. Trying single newline split...")
        paragraphs = [p.strip() for p in text.split("\n") if p.strip()]

    if len(paragraphs) < 4:
        print(f"  FATAL: Corpus has fewer than 4 paragraphs ({len(paragraphs)}). Too small.")
        sys.exit(1)

    midpoint = len(paragraphs) // 2
    half_a = "\n\n".join(paragraphs[:midpoint])
    half_b = "\n\n".join(paragraphs[midpoint:])
    full = "\n\n".join(paragraphs)

    # Write splits
    paths = {
        "half_a": os.path.join(output_dir, "half_a.txt"),
        "half_b": os.path.join(output_dir, "half_b.txt"),
        "full": os.path.join(output_dir, "full.txt"),
    }

    with open(paths["half_a"], "w", encoding="utf-8") as f:
        f.write(half_a)
    with open(paths["half_b"], "w", encoding="utf-8") as f:
        f.write(half_b)
    with open(paths["full"], "w", encoding="utf-8") as f:
        f.write(full)

    # Compute stats
    def approx_tokens(text: str) -> int:
        """Approximate token count (1 token ≈ 4 characters)."""
        return len(text) // 4

    stats = {
        "total_paragraphs": len(paragraphs),
        "half_a_paragraphs": midpoint,
        "half_b_paragraphs": len(paragraphs) - midpoint,
        "half_a_chars": len(half_a),
        "half_b_chars": len(half_b),
        "full_chars": len(full),
        "half_a_tokens": approx_tokens(half_a),
        "half_b_tokens": approx_tokens(half_b),
        "full_tokens": approx_tokens(full),
        "half_a_md5": hashlib.md5(half_a.encode()).hexdigest()[:8],
        "half_b_md5": hashlib.md5(half_b.encode()).hexdigest()[:8],
        "full_md5": hashlib.md5(full.encode()).hexdigest()[:8],
    }

    return stats


def validate_splits(stats: dict) -> bool:
    """Validate that A + B == Full."""
    ok = True

    if stats["half_a_paragraphs"] + stats["half_b_paragraphs"] != stats["total_paragraphs"]:
        print("  ✗ VALIDATION FAILED: paragraph counts don't add up!")
        ok = False

    # Character count validation (allow small rounding from join separators)
    expected_chars = stats["half_a_chars"] + stats["half_b_chars"] + 2  # +2 for the \n\n separator
    actual_chars = stats["full_chars"]
    if abs(expected_chars - actual_chars) > 10:
        print(f"  ✗ VALIDATION FAILED: char counts off by {abs(expected_chars - actual_chars)}")
        ok = False

    if ok:
        print("  ✓ Validation passed: A + B == Full")

    return ok


def main():
    print("=" * 50)
    print("  Step 1: Dataset Preparation")
    print("=" * 50)
    print()

    # Download
    print("[1/3] Downloading corpus...")
    download_book(BOOK_URL, BOOK_PATH)

    # Split
    print()
    print("[2/3] Splitting corpus...")
    stats = split_corpus(BOOK_PATH, SPLITS_DIR)

    # Report
    print()
    print("[3/3] Summary:")
    print(f"  {'':30s} {'Paragraphs':>12s} {'~Tokens':>10s} {'Chars':>10s} {'MD5':>10s}")
    print(f"  {'-'*72}")
    print(f"  {'Half A (first half)':30s} {stats['half_a_paragraphs']:>12d} {stats['half_a_tokens']:>10,d} {stats['half_a_chars']:>10,d} {stats['half_a_md5']:>10s}")
    print(f"  {'Half B (second half)':30s} {stats['half_b_paragraphs']:>12d} {stats['half_b_tokens']:>10,d} {stats['half_b_chars']:>10,d} {stats['half_b_md5']:>10s}")
    print(f"  {'Full (complete)':30s} {stats['total_paragraphs']:>12d} {stats['full_tokens']:>10,d} {stats['full_chars']:>10,d} {stats['full_md5']:>10s}")
    print()

    # Validate
    validate_splits(stats)

    print()
    print(f"  ✓ Files written to: {SPLITS_DIR}")
    print()

    return stats


if __name__ == "__main__":
    main()
