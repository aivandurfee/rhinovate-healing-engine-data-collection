#!/usr/bin/env python3
"""
American Society of Plastic Surgeons (ASPS) download script.

Scrape rhinoplasty / before-after data from ASPS (e.g. before/after galleries).
To be implemented as a separate scraper from the Reddit pipeline.

Usage (when implemented):
  python scripts/asps_download.py [options]

Data can be written to e.g. data/raw_downloads/asps/ and then fed into
the same sort/align pipeline as Reddit data if desired.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    print("ASPS download not yet implemented.")
    print("This script will scrape American Society of Plastic Surgeons when ready.")
    sys.exit(1)


if __name__ == "__main__":
    main()
