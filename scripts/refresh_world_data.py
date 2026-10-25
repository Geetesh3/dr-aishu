"""Refresh the cached real-world health statistics used on the Insights page.

Writes `data/external/world/world_health.json` from the WHO Global Health
Observatory and World Bank Open Data APIs. Run it when you want newer figures;
the app itself only ever reads the cache, so a deploy with no network still
works and an upstream outage cannot break a page.

Run:
    python scripts/refresh_world_data.py
    python scripts/refresh_world_data.py --country USA
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import worlddata  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--country", default=worlddata.DEFAULT_COUNTRY,
                    help="ISO-3 country code (default: IND)")
    args = ap.parse_args(argv)

    print(f"Refreshing world health data for {args.country} ...")
    payload = worlddata.refresh(args.country)

    ok = 0
    for group in ("who", "world_bank"):
        for item in payload[group]:
            latest = item.get("latest")
            if latest:
                ok += 1
                print(f"  {item['label'][:58]:<58} {latest['year']}  {latest['value']}")
            else:
                print(f"  {item['label'][:58]:<58} no data"
                      + (f" ({item['error']})" if item.get("error") else ""))
    print(f"\n{ok} indicator(s) with data. Cache: {worlddata.CACHE.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
