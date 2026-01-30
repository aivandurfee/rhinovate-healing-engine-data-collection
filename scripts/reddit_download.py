"""
Reddit download script.
Runs gallery-dl (with gallery-dl.conf) to download from Reddit,
then runs the sort pipeline to organize into Before / After / Healing.

Usage:
  python scripts/reddit_download.py                    # rhinoplasty-only search (default), limit 100
  python scripts/reddit_download.py --no-search        # use r/PlasticSurgery/new (all plastic surgery; lip fillers, etc.)
  python scripts/reddit_download.py --limit 50         # cap 50 posts per query
  python scripts/reddit_download.py --full             # no limit
  python scripts/reddit_download.py --cookies-from-browser chrome  # use browser cookies (if Reddit rate-limits)
  python scripts/reddit_download.py <URL>              # single post or custom Reddit URL

Examples:
  python scripts/reddit_download.py
  python scripts/reddit_download.py --no-search --limit 100
  python scripts/reddit_download.py "https://www.reddit.com/r/PlasticSurgery/comments/abc123/title/"
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from urllib.parse import quote_plus

# Project root (parent of scripts/)
ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "gallery-dl.conf"

BASE = "https://www.reddit.com/r/PlasticSurgery/search"
SUBREDDIT_NEW = "https://www.reddit.com/r/PlasticSurgery/new/"

RHINO_SEARCH_QUERIES = [
    "rhinoplasty",
    "nose job",
    "rhinoplasty results",
    "nose job results",
]


def _rhino_search_urls() -> list[str]:
    out = []
    for q in RHINO_SEARCH_QUERIES:
        param = quote_plus(q)
        out.append(f"{BASE}/?q={param}&restrict_sr=1&sort=new")
    return out


def _parse_args() -> tuple[list[str], int | None, bool, bool, str | None]:
    """Return (urls, limit or None, use_full, no_search, cookies_browser or None)."""
    args = sys.argv[1:]
    url_override: str | None = None
    limit: int | None = 100
    use_full = False
    no_search = False
    cookies_browser: str | None = None  # default off; use --cookies-from-browser chrome if needed

    i = 0
    while i < len(args):
        a = args[i]
        if a == "--limit" and i + 1 < len(args):
            try:
                limit = max(1, int(args[i + 1]))
            except ValueError:
                limit = 100
            i += 2
            continue
        if a == "--full":
            use_full = True
            limit = None
            i += 1
            continue
        if a == "--no-search":
            no_search = True
            i += 1
            continue
        if a == "--no-cookies":
            cookies_browser = None
            i += 1
            continue
        if a == "--cookies-from-browser" and i + 1 < len(args):
            cookies_browser = args[i + 1].strip().lower()
            i += 2
            continue
        if not a.startswith("-"):
            url_override = a.strip()
            i += 1
            break
        i += 1

    if url_override:
        return [url_override], None, True, False, cookies_browser
    if no_search:
        urls = [SUBREDDIT_NEW]
        return urls, limit, use_full, True, cookies_browser
    urls = _rhino_search_urls()
    if use_full:
        return urls, None, True, False, cookies_browser
    return urls, limit, False, False, cookies_browser


def main() -> None:
    urls, limit, _, no_search, cookies_browser = _parse_args()

    os.chdir(ROOT)
    print(f"Working directory: {ROOT}")
    print(f"Config: {CONFIG_PATH}")
    if len(urls) == 1 and urls[0] == SUBREDDIT_NEW:
        print(f"Target: {SUBREDDIT_NEW} (subreddit /new, no search)")
    elif len(urls) == 1 and not urls[0].startswith(BASE):
        print(f"Target: {urls[0]}")
    else:
        print(f"Target: rhinoplasty search ({len(urls)} queries)")
        for u in urls:
            print(f"  - {u}")
    if limit is not None:
        print(f"Limit: --range 1-{limit} (per query)")
    if cookies_browser:
        print(f"Cookies: --cookies-from-browser {cookies_browser}")
    else:
        print("Cookies: none")
    sys.stdout.flush()

    cmd = [
        "gallery-dl",
        "--config", str(CONFIG_PATH),
        "--filter", "extension in ('jpg','jpeg','png','webp','gif')",
    ]
    if cookies_browser:
        cmd.extend(["--cookies-from-browser", cookies_browser])
    if limit is not None:
        cmd.extend(["--range", f"1-{limit}"])
    cmd.extend(urls)

    print("\nLaunching gallery-dl...")
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")

    if r.returncode != 0:
        print("gallery-dl exited with an error. Output below:")
        if r.stdout:
            print("-- stdout --")
            print(r.stdout)
        if r.stderr:
            print("-- stderr --")
            print(r.stderr)
        err = (r.stderr or "") + (r.stdout or "")
        if "Access is denied" in err and "cookies" in err.lower():
            print("Tip: Cookie extraction failed. Run with --no-cookies and try again.")
        elif "KeyError" in err or "'data'" in err:
            print("Tip: Reddit API change or auth issue. Try --no-search and/or --no-cookies.")
        else:
            print("Tip: Try --no-search (subreddit /new), --no-cookies, or --cookies-from-browser firefox.")
        print("Continuing to sort existing data.\n")

    print("Running sort pipeline...")
    sys.path.insert(0, str(ROOT / "scripts"))
    from sort_healing_data import categorize_images

    categorize_images()
    print("\nAll done.")
    if r.returncode != 0:
        sys.exit(r.returncode)


if __name__ == "__main__":
    main()
