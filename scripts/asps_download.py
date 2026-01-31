#!/usr/bin/env python3
"""
American Society of Plastic Surgeons (ASPS) – Rhinoplasty Before/After Scraper.

Production-ready Selenium scraper that:
  1. Navigates to the ASPS rhinoplasty photo gallery
  2. Collects all case links (with pagination)
  3. For each case, extracts Before and After image URLs and downloads them
  4. Writes to the same structure as the Reddit pipeline for downstream ML:
     data/clean_dataset/asps_case_{id}/Before/  and  .../After/

Designed for high-resolution, standardized medical photography. Requires:
  - Python 3.9+
  - selenium, requests (pip install -r requirements.txt)
  - Chrome browser + ChromeDriver (or --browser firefox / edge). Install ChromeDriver:
    https://chromedriver.chromium.org/ or use: chromedriver-autoinstaller / webdriver-manager

Usage:
  python scripts/asps_download.py                    # scrape all pages, default limit
  python scripts/asps_download.py --max-cases 50      # stop after 50 cases
  python scripts/asps_download.py --max-pages 2       # only first 2 gallery pages
  python scripts/asps_download.py --headless          # run browser headless
  python scripts/asps_download.py --dry-run          # collect URLs only, no downloads
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

from selenium.webdriver.common.by import By
from selenium.common.exceptions import (
    InvalidSessionIdException,
    WebDriverException,
)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
CLEAN_DIR = ROOT / "data" / "clean_dataset"
RAW_ASPS_DIR = ROOT / "data" / "raw_downloads" / "asps"

# ASPS rhinoplasty gallery (paginated)
GALLERY_BASE = "https://www.plasticsurgery.org/photo-gallery/procedure/rhinoplasty"
CASE_URL_PATTERN = re.compile(
    r"/photo-gallery/procedure/rhinoplasty/case/(\d+)",
    re.I,
)

# Image host (before/after images are served from www1 subdomain)
# Filename convention observed: {case_id}-{img_id}b_* = before, *a_* = after
IMAGE_BASE = "https://www1.plasticsurgery.org/include/images/photogallery/cases"

# Politeness and robustness
REQUEST_DELAY_SEC = 1.0
PAGINATION_DELAY_SEC = 0.8
CASE_PAGE_DELAY_SEC = 0.5
MAX_RETRIES = 3
RETRY_DELAY_SEC = 2.0
PAGE_LOAD_TIMEOUT_SEC = 25
IMPLICIT_WAIT_SEC = 10

# User-Agent for HTTP requests (avoid blocks)
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scrape ASPS rhinoplasty Before/After photos into clean_dataset."
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Stop after this many cases (default: no limit).",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Stop after this many gallery pages (default: all).",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run browser in headless mode.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only collect case URLs; do not download images.",
    )
    parser.add_argument(
        "--browser",
        choices=("chrome", "firefox", "edge"),
        default="chrome",
        help="Browser for Selenium (default: chrome).",
    )
    return parser.parse_args()


def _create_driver(headless: bool, browser: str):
    """Create and return a configured Selenium WebDriver."""
    try:
        if browser == "chrome":
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options
            from selenium.webdriver.chrome.service import Service

            opts = Options()
            if headless:
                opts.add_argument("--headless=new")
            opts.add_argument("--no-sandbox")
            opts.add_argument("--disable-dev-shm-usage")
            opts.add_argument("--disable-gpu")
            opts.add_argument(f"--user-agent={USER_AGENT}")
            opts.add_argument("--window-size=1920,1080")
            driver = webdriver.Chrome(options=opts)
        elif browser == "firefox":
            from selenium import webdriver
            from selenium.webdriver.firefox.options import Options

            opts = Options()
            if headless:
                opts.add_argument("--headless")
            opts.set_preference("general.useragent.override", USER_AGENT)
            driver = webdriver.Firefox(options=opts)
        elif browser == "edge":
            from selenium import webdriver
            from selenium.webdriver.edge.options import Options

            opts = Options()
            if headless:
                opts.add_argument("--headless")
            opts.add_argument(f"--user-agent={USER_AGENT}")
            driver = webdriver.Edge(options=opts)
        else:
            raise ValueError(f"Unsupported browser: {browser}")
    except Exception as e:
        print(f"Failed to create WebDriver: {e}", file=sys.stderr)
        print("Install Chrome/ChromeDriver or use --browser firefox with GeckoDriver.", file=sys.stderr)
        sys.exit(1)

    driver.set_page_load_timeout(PAGE_LOAD_TIMEOUT_SEC)
    driver.implicitly_wait(IMPLICIT_WAIT_SEC)
    return driver


def _collect_case_links(driver, max_pages: int | None) -> list[tuple[str, str]]:
    """
    Navigate gallery pages and collect (case_url, case_id) for each rhinoplasty case.
    Returns list of (url, case_id) with case_id stripped from path.
    """
    seen_ids: set[str] = set()
    results: list[tuple[str, str]] = []
    page_num = 1

    while True:
        if max_pages is not None and page_num > max_pages:
            break
        url = GALLERY_BASE if page_num == 1 else f"{GALLERY_BASE}/page/{page_num}"
        print(f"  Gallery page {page_num}: {url}")
        try:
            driver.get(url)
            time.sleep(PAGINATION_DELAY_SEC)
        except (InvalidSessionIdException, WebDriverException) as e:
            print(f"  Browser closed or disconnected: {e}", file=sys.stderr)
            print(f"  Returning {len(results)} case links collected so far.")
            return results
        except Exception as e:
            print(f"  Failed to load page: {e}", file=sys.stderr)
            break

        # Find all links that match /procedure/rhinoplasty/case/{id}
        try:
            links = driver.find_elements(By.TAG_NAME, "a")
        except (InvalidSessionIdException, WebDriverException) as e:
            print(f"  Browser closed or disconnected: {e}", file=sys.stderr)
            print(f"  Returning {len(results)} case links collected so far.")
            return results

        page_ids: set[str] = set()
        for a in links:
            try:
                href = a.get_attribute("href") or ""
            except (InvalidSessionIdException, WebDriverException) as e:
                print(f"  Browser closed or disconnected: {e}", file=sys.stderr)
                print(f"  Returning {len(results)} case links collected so far.")
                return results
            m = CASE_URL_PATTERN.search(href)
            if m:
                case_id = m.group(1)
                if case_id not in seen_ids:
                    seen_ids.add(case_id)
                    page_ids.add(case_id)
                    full_url = urljoin(GALLERY_BASE, href) if not href.startswith("http") else href
                    results.append((full_url, case_id))

        if not page_ids:
            # No new cases on this page; stop pagination
            break
        page_num += 1
        time.sleep(REQUEST_DELAY_SEC)

    return results


def _normalize_image_url(url: str, prefer_full_res: bool = True) -> str:
    """
    Prefer full-resolution image: ASPS often serves _scaled.jpg; try without _scaled.
    """
    if not url or not url.strip():
        return url
    u = url.strip()
    if prefer_full_res and "_scaled" in u:
        u = u.replace("_scaled", "")
    return u


def _download_image(url: str, dest: Path, session) -> bool:
    """Download a single image to dest. Tries full-res first, then _scaled. Returns True on success."""
    candidates = [_normalize_image_url(url, prefer_full_res=True), url]
    if url not in candidates:
        candidates.append(url)
    last_err = None
    for try_url in candidates:
        for attempt in range(MAX_RETRIES):
            try:
                r = session.get(try_url, timeout=30, headers={"User-Agent": USER_AGENT})
                r.raise_for_status()
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(r.content)
                return True
            except Exception as e:
                last_err = e
                # If 404 and we tried full-res, fall back to next candidate (e.g. _scaled)
                if getattr(e, "response", None) and getattr(e.response, "status_code", None) == 404:
                    break
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY_SEC)
        else:
            continue
        continue
    print(f"    Download failed: {url[:80]}... – {last_err}", file=sys.stderr)
    return False


def _extract_before_after_urls(driver) -> tuple[list[str], list[str]]:
    """
    On a case page, find all image URLs and classify into before (b) and after (a)
    based on ASPS filename convention: *b_scaled.jpg / *b.jpg = before, *a_* = after.
    Returns (before_urls, after_urls).
    """
    before_urls: list[str] = []
    after_urls: list[str] = []

    # Prefer data-src or full-size src; fallback to src
    imgs = driver.find_elements(By.TAG_NAME, "img")
    for img in imgs:
        src = img.get_attribute("data-src") or img.get_attribute("src") or ""
        if "photogallery" not in src or "plasticsurgery" not in src:
            continue
        # Normalize: ensure we use https and full host
        if src.startswith("//"):
            src = "https:" + src
        if "www1.plasticsurgery.org" not in src and "plasticsurgery.org" in src:
            src = src.replace("www.plasticsurgery.org", "www1.plasticsurgery.org", 1)
        # ASPS convention: ...-101964b_scaled.jpg = before, ...-101964a_scaled.jpg = after
        if re.search(r"[_-](\d+)b[_.]", src, re.I):
            before_urls.append(src)
        elif re.search(r"[_-](\d+)a[_.]", src, re.I):
            after_urls.append(src)

    # Also check <a href="..."> linking to image URLs (e.g. lightbox links)
    for a in driver.find_elements(By.TAG_NAME, "a"):
        href = a.get_attribute("href") or ""
        if "photogallery" not in href or "plasticsurgery" not in href:
            continue
        if href.startswith("//"):
            href = "https:" + href
        if re.search(r"[_-](\d+)b[_.]", href, re.I):
            before_urls.append(href)
        elif re.search(r"[_-](\d+)a[_.]", href, re.I):
            after_urls.append(href)

    # Deduplicate while preserving order
    before_urls = list(dict.fromkeys(before_urls))
    after_urls = list(dict.fromkeys(after_urls))
    return before_urls, after_urls


def _scrape_case(
    driver,
    case_url: str,
    case_id: str,
    session,
    clean_dir: Path,
    raw_dir: Path,
    dry_run: bool,
) -> bool:
    """
    Open case page, extract before/after image URLs, download to clean_dataset and raw.
    Returns True if at least one image was saved (or dry_run succeeded).
    """
    try:
        driver.get(case_url)
        time.sleep(CASE_PAGE_DELAY_SEC)
    except Exception as e:
        print(f"  Case {case_id} failed to load: {e}", file=sys.stderr)
        return False

    before_urls, after_urls = _extract_before_after_urls(driver)
    if not before_urls and not after_urls:
        print(f"  Case {case_id}: no before/after images found on page.")
        return False

    if dry_run:
        print(f"  Case {case_id}: {len(before_urls)} before, {len(after_urls)} after (dry-run, skip download)")
        return True

    # Downstream ML structure: clean_dataset/asps_case_{id}/Before/ and After/
    patient_id = f"asps_case_{case_id}"
    clean_before = clean_dir / patient_id / "Before"
    clean_after = clean_dir / patient_id / "After"
    raw_before = raw_dir / case_id / "Before"
    raw_after = raw_dir / case_id / "After"

    # Download each image once to clean path, then copy to raw (avoids double HTTP fetch)
    saved = 0
    for i, url in enumerate(before_urls):
        ext = ".jpg" if ".jpg" in url.split("?")[0] else ".png"
        name = f"{(i+1):02d}_before{ext}"
        clean_path = clean_before / name
        raw_path = raw_before / name
        if _download_image(url, clean_path, session):
            saved += 1
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(clean_path, raw_path)
    for i, url in enumerate(after_urls):
        ext = ".jpg" if ".jpg" in url.split("?")[0] else ".png"
        name = f"{(i+1):02d}_after{ext}"
        clean_path = clean_after / name
        raw_path = raw_after / name
        if _download_image(url, clean_path, session):
            saved += 1
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(clean_path, raw_path)

    if saved:
        print(f"  Case {case_id}: saved {saved} images -> {patient_id}/Before|After")
    return saved > 0


def main() -> None:
    args = _parse_args()
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    RAW_ASPS_DIR.mkdir(parents=True, exist_ok=True)

    print("ASPS Rhinoplasty Before/After Scraper")
    print(f"  Output (ML): {CLEAN_DIR}")
    print(f"  Raw copy:   {RAW_ASPS_DIR}")
    if args.dry_run:
        print("  Mode: dry-run (no downloads)")
    print()

    driver = _create_driver(args.headless, args.browser)
    try:
        print("Collecting case links from gallery...")
        case_list = _collect_case_links(driver, args.max_pages)
        if args.max_cases is not None:
            case_list = case_list[: args.max_cases]
        print(f"Found {len(case_list)} unique cases.")
        if not case_list:
            print("Nothing to scrape.")
            return

        # If browser died during link collection, driver is invalid; recreate for download phase
        try:
            _ = driver.current_url
        except (InvalidSessionIdException, WebDriverException):
            print("Recreating browser for download phase (previous session closed).")
            try:
                driver.quit()
            except Exception:
                pass
            driver = _create_driver(args.headless, args.browser)

        try:
            import requests
        except ImportError:
            print("Install 'requests' for image downloads: pip install requests", file=sys.stderr)
            sys.exit(1)
        session = requests.Session()
        session.headers["User-Agent"] = USER_AGENT

        success = 0
        for case_url, case_id in case_list:
            try:
                if _scrape_case(
                    driver,
                    case_url,
                    case_id,
                    session,
                    CLEAN_DIR,
                    RAW_ASPS_DIR,
                    args.dry_run,
                ):
                    success += 1
            except (InvalidSessionIdException, WebDriverException) as e:
                print(f"\nBrowser closed or disconnected: {e}", file=sys.stderr)
                print(f"Processed {success}/{len(case_list)} cases before browser died.")
                break
            time.sleep(REQUEST_DELAY_SEC)

        print(f"\nDone. Processed {success}/{len(case_list)} cases with images.")
    finally:
        try:
            driver.quit()
        except Exception:
            pass


if __name__ == "__main__":
    main()
