"""
Usage: uv run python scripts/probe_mapillary.py
"""

from __future__ import annotations

import os

import requests

from comp4330sef.config import REPO_ROOT

URL = "https://graph.mapillary.com/images"
TINY = "114.1600,22.2800,114.1650,22.2850"   # ~500 m box in Central
HONG = "113.8300,22.1500,114.4400,22.5600"   # the full HK bbox

CANDIDATE_FIELDS = (
    "captured_at",
    "geometry",
    "computed_geometry",
    "compass_angle",
    "sequence",
    "is_pano",
    "width",
    "height",
    "creator",
    "quality_score",
    "thumb_original_url",
    "thumb_2048_url",
    "thumb_1024_url",
)


def load_token() -> str:
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
    token = os.environ.get("MAPILLARY_TOKEN", "").strip()
    if not token:
        raise SystemExit("MAPILLARY_TOKEN missing from .env")
    return token


def probe(token: str, label: str, **params) -> None:
    params["access_token"] = token
    try:
        response = requests.get(URL, params=params, timeout=90)
    except Exception as exc:
        print(f"{label:<36} EXC    {exc}")
        return
    body = " ".join(response.text.split())[:280]
    print(f"{label:<36} {response.status_code:<4} {body}")


def main() -> int:
    token = load_token()
    print(f"token : {token[:8]}... ({len(token)} chars)\n")

    print("--- bbox / limit ---")
    probe(token, "tiny bbox, fields=id, limit=1", fields="id", bbox=TINY, limit=1)
    probe(token, "tiny bbox, fields=id, limit=200", fields="id", bbox=TINY, limit=200)
    probe(token, "HK bbox, fields=id, limit=1", fields="id", bbox=HONG, limit=1)
    probe(token, "HK bbox, fields=id, limit=200", fields="id", bbox=HONG, limit=200)

    print("\n--- one field at a time (tiny bbox) ---")
    for field in CANDIDATE_FIELDS:
        probe(token, f"tiny, id+{field}", fields=f"id,{field}", bbox=TINY, limit=1)

    print("\n--- 0.12 deg tiles across HK ---")
    for index, west in enumerate((113.83, 113.95, 114.07, 114.19)):
        bbox = f"{west:.4f},22.15,{west + 0.12:.4f},22.30"
        probe(token, f"tile {index}", fields="id,captured_at", bbox=bbox, limit=50)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())