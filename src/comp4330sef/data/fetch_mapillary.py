"""
Usage:
    uv run python -m comp4330sef.data.fetch_mapillary --dry-run
    uv run python -m comp4330sef.data.fetch_mapillary --target 260 --candidates 600 --sheets
    uv run python -m comp4330sef.data.fetch_mapillary --sheets-only
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import subprocess
import time
from collections import defaultdict
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
from PIL import Image

from comp4330sef.config import REPO_ROOT

API_IMAGES = "https://graph.mapillary.com/images"
LICENSE = "CC BY-SA 4.0"
HK_BBOX = (113.83, 22.15, 114.44, 22.56)
URBAN_REGIONS: dict[str, tuple[float, float, float, float]] = {
    # --- Hong Kong Island ---
    "central-sheung-wan": (114.1450, 22.2790, 114.1720, 22.2900),
    "wan-chai-causeway-bay": (114.1700, 22.2750, 114.1920, 22.2850),
    "north-point": (114.1900, 22.2830, 114.2060, 22.2950),
    "quarry-bay-taikoo": (114.2060, 22.2830, 114.2200, 22.2950),
    "kennedy-town-sai-ying-pun": (114.1200, 22.2800, 114.1480, 22.2920),
    "chai-wan": (114.2200, 22.2600, 114.2450, 22.2750),
    "aberdeen-wong-chuk-hang": (114.1550, 22.2450, 114.1750, 22.2600),
    # --- Kowloon ---
    "tsim-sha-tsui-jordan": (114.1650, 22.2950, 114.1800, 22.3050),
    "mong-kok-yau-ma-tei": (114.1600, 22.3050, 114.1780, 22.3200),
    "sham-shui-po-cheung-sha-wan": (114.1500, 22.3250, 114.1700, 22.3400),
    "hung-hom-to-kwa-wan": (114.1800, 22.3050, 114.1950, 22.3200),
    "kwun-tong-ngau-tau-kok": (114.2150, 22.3100, 114.2350, 22.3250),
    "wong-tai-sin-san-po-kong": (114.1900, 22.3350, 114.2050, 22.3450),
    # --- New Territories town centres ---
    "tsuen-wan": (114.1050, 22.3650, 114.1250, 22.3780),
    "kwai-chung": (114.1200, 22.3550, 114.1400, 22.3680),
    "sha-tin": (114.1800, 22.3750, 114.2000, 22.3900),
    "tai-po": (114.1600, 22.4400, 114.1750, 22.4550),
    "yuen-long": (114.0250, 22.4400, 114.0450, 22.4500),
    "tuen-mun": (113.9700, 22.3900, 113.9900, 22.4050),
    "sheung-shui-fanling": (114.1200, 22.4950, 114.1400, 22.5100),
    "tseung-kwan-o": (114.2500, 22.3050, 114.2700, 22.3200),
}
FIELDS = (
    "id,captured_at,geometry,computed_geometry,compass_angle,sequence,is_pano,"
    "width,height,creator,quality_score,thumb_original_url,thumb_2048_url,thumb_1024_url"
)

MAX_API_BBOX_AREA = 0.010  # Mapillary hard limit, square degrees
DEFAULT_TILE_AREA = 0.006

RAW_DIR = REPO_ROOT / "data" / "raw"
IMAGES_DIR = REPO_ROOT / "data" / "images"
ANNOTATIONS_DIR = REPO_ROOT / "data" / "annotations"
TRIAGE_DIR = REPO_ROOT / "data" / "triage"
CACHE_DIR = REPO_ROOT / ".cache"
CACHE_FILE = CACHE_DIR / "mapillary_records.json"

CSV_COLUMNS = (
    "image_id",
    "file_name",
    "mapillary_id",
    "captured_at_utc",
    "lat",
    "lon",
    "region",
    "sequence",
    "compass_angle",
    "creator",
    "license",
    "attribution",
    "original_url",
    "original_width",
    "original_height",
    "width",
    "height",
    "max_side",
    "sha256",
    "dhash",
    "scale_bucket",
    "occlusion_level",
    "illumination",
    "text_density",
    "keep",
    "note",
)


class RequestFailed(RuntimeError):
    """The Mapillary API kept failing after every retry."""


# HTTP

def _session() -> Any:
    try:
        import requests
    except ImportError as exc:
        raise SystemExit('requests is missing: run "uv add requests"') from exc
    session = requests.Session()
    session.headers.update({"User-Agent": "comp4330sef-phase1/1.0"})
    return session


def _request(session: Any, url: str, params: dict[str, Any] | None = None, tries: int = 5) -> Any:
    delay = 2.0
    last_error = ""
    for attempt in range(1, tries + 1):
        try:
            response = session.get(url, params=params, timeout=120)
        except Exception as exc:
            last_error = f"network error: {exc}"
            if attempt == tries:
                break
            time.sleep(min(delay * (0.5 + random.random()), 60))
            delay *= 2
            continue

        if response.status_code == 200:
            return response

        body = " ".join(response.text.split())[:300]

        if response.status_code == 500 and "reduce the amount of data" in body.lower():
            raise RequestFailed(f"HTTP 500: {body}")

        if response.status_code in (429, 500, 502, 503, 504):
            last_error = f"HTTP {response.status_code}: {body}"
            if attempt == tries:
                break
            wait = float(response.headers.get("Retry-After", delay))
            time.sleep(min(wait * (0.5 + random.random()), 60))
            delay *= 2
            continue

        raise SystemExit(
            f"Mapillary returned HTTP {response.status_code} for {response.url}\n"
            f"{response.text[:400]}"
        )

    raise RequestFailed(f"giving up after {tries} attempts: {last_error}")


def _token() -> str:
    try:
        from dotenv import load_dotenv
    except ImportError as exc:
        raise SystemExit('python-dotenv is missing: run "uv sync"') from exc
    load_dotenv(REPO_ROOT / ".env")
    token = os.environ.get("MAPILLARY_TOKEN", "").strip()
    if not token:
        raise SystemExit(
            "MAPILLARY_TOKEN is not set. Create .env in the repo root with:\n"
            "  MAPILLARY_TOKEN=MLY|your_client_id|your_client_secret"
        )
    return token


# metadata

def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        for key in ("username", "id", "name"):
            if value.get(key):
                return str(value[key])
        return ""
    return str(value)


def _coords(record: dict[str, Any]) -> tuple[float, float] | None:
    geometry = record.get("geometry") or record.get("computed_geometry") or {}
    coordinates = (geometry or {}).get("coordinates") or []
    if len(coordinates) >= 2:
        return float(coordinates[0]), float(coordinates[1])
    return None


def _captured_ms(record: dict[str, Any]) -> int | None:
    value = record.get("captured_at")
    return int(value) if value is not None else None


def _captured_iso(milliseconds: int | None) -> str:
    if milliseconds is None:
        return ""
    return datetime.fromtimestamp(milliseconds / 1000.0, tz=timezone.utc).isoformat(timespec="seconds")


def _thumb(record: dict[str, Any], size: str) -> str:
    order = {
        "2048": ("thumb_2048_url", "thumb_original_url", "thumb_1024_url"),
        "original": ("thumb_original_url", "thumb_2048_url", "thumb_1024_url"),
        "1024": ("thumb_1024_url", "thumb_2048_url", "thumb_original_url"),
    }[size]
    for key in order:
        value = record.get(key)
        if value:
            return str(value)
    return ""


def _sequence(record: dict[str, Any]) -> str:
    sequence = _text(record.get("sequence"))
    return sequence or f"single:{record.get('id')}"


def _normalise_record(record: dict[str, Any], size: str) -> dict[str, Any] | None:
    mapillary_id = _text(record.get("id"))
    coords = _coords(record)
    thumbnail = _thumb(record, size)
    if not mapillary_id or coords is None or not thumbnail:
        return None
    if record.get("is_pano"):
        return None
    milliseconds = _captured_ms(record)
    return {
        "mapillary_id": mapillary_id,
        "captured_ms": milliseconds,
        "captured_at_utc": _captured_iso(milliseconds),
        "lat": round(coords[1], 7),
        "lon": round(coords[0], 7),
        "sequence": _sequence(record),
        "compass_angle": _text(record.get("compass_angle")),
        "creator": _text(record.get("creator")),
        "thumb": thumbnail,
    }


# search

def tile_bboxes(bbox: tuple[float, ...], cols: int, rows: int) -> Iterator[tuple[float, ...]]:
    west, south, east, north = bbox
    dx = (east - west) / cols
    dy = (north - south) / rows
    for row in range(rows):
        for col in range(cols):
            yield (
                west + col * dx,
                south + row * dy,
                west + (col + 1) * dx,
                south + (row + 1) * dy,
            )


def choose_grid(bbox: tuple[float, ...], max_tile_area: float) -> tuple[int, int]:
    """Smallest grid such that every tile stays under the API area limit."""
    west, south, east, north = bbox
    width = east - west
    height = north - south
    if width <= 0 or height <= 0:
        raise SystemExit("bbox must satisfy west < east and south < north")
    if width * height <= max_tile_area:
        return 1, 1
    tile_count = math.ceil(width * height / max_tile_area)
    rows = max(1, math.ceil(math.sqrt(tile_count * height / width)))
    cols = max(1, math.ceil(tile_count / rows))
    while (width / cols) * (height / rows) > max_tile_area:
        cols += 1
    return cols, rows


def _regions_from_args(args: argparse.Namespace) -> list[tuple[str, tuple[float, ...]]]:
    if args.bbox:
        return [("custom", tuple(float(value) for value in args.bbox))]
    if args.preset == "urban":
        return [(name, bbox) for name, bbox in URBAN_REGIONS.items()]
    return [("hk", HK_BBOX)]


def search_tile(
    session: Any,
    token: str,
    bbox: tuple[float, ...],
    limit: int,
    page_size: int,
    pause: float,
) -> list[dict[str, Any]]:
    params = {
        "access_token": token,
        "fields": FIELDS,
        "bbox": ",".join(f"{value:.6f}" for value in bbox),
        "limit": min(page_size, 2000),
    }
    records: list[dict[str, Any]] = []
    url: str | None = API_IMAGES
    query: dict[str, Any] | None = params
    while url and len(records) < limit:
        try:
            payload = _request(session, url, query).json()
        except RequestFailed:
            if not records:
                raise
            print("    [warn] pagination stopped early; keeping partial tile results")
            break
        records.extend(payload.get("data") or [])
        url = (payload.get("paging") or {}).get("next")
        query = None
        if url and len(records) < limit:
            time.sleep(pause)
    return records[:limit]


# dedupe

def dhash(path: Path, size: int = 8) -> int:
    with Image.open(path) as handle:
        gray = handle.convert("L").resize((size + 1, size), Image.LANCZOS)
    pixels = np.asarray(gray, dtype=np.int16)
    bits = (pixels[:, 1:] > pixels[:, :-1]).reshape(-1)
    value = 0
    for bit in bits:
        value = (value << 1) | int(bool(bit))
    return value


def hamming(left: int, right: int) -> int:
    return bin(left ^ right).count("1")


def spaced(items: Iterable[dict[str, Any]], max_count: int, min_gap_seconds: int) -> list[dict[str, Any]]:
    """Keep at most max_count frames from one sequence, >= min_gap_seconds apart."""
    chosen: list[dict[str, Any]] = []
    last_ms: int | None = None
    for item in items:
        milliseconds = item.get("captured_ms")
        if (
            last_ms is not None
            and milliseconds is not None
            and milliseconds - last_ms < min_gap_seconds * 1000
        ):
            continue
        chosen.append(item)
        last_ms = milliseconds
        if len(chosen) >= max_count:
            break
    return chosen


def round_robin(groups: Iterable[list[dict[str, Any]]]) -> Iterator[dict[str, Any]]:
    queues = [list(group) for group in groups if group]
    while queues:
        remaining = []
        for queue in queues:
            yield queue.pop(0)
            if queue:
                remaining.append(queue)
        queues = remaining


# downloads

def download(session: Any, url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    response = _request(session, url)
    content_type = response.headers.get("Content-Type", "")
    if "image" not in content_type:
        raise RuntimeError(f"unexpected Content-Type {content_type!r} for {url}")
    temporary.write_bytes(response.content)
    temporary.replace(destination)


def normalise(raw_path: Path, out_path: Path, max_side: int) -> tuple[int, int]:
    with Image.open(raw_path) as handle:
        image = handle.convert("RGB")
        width, height = image.size
        if max_side and max(width, height) > max_side:
            scale = max_side / max(width, height)
            image = image.resize(
                (max(1, round(width * scale)), max(1, round(height * scale))), Image.LANCZOS
            )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(out_path, "JPEG", quality=95, optimize=True)
        return image.size


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# sheets

def write_sheets(rows: list[dict[str, str]], per_sheet: int, columns: int) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR / "matplotlib"))
    (CACHE_DIR / "matplotlib").mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    TRIAGE_DIR.mkdir(parents=True, exist_ok=True)
    rows_per_sheet = max(1, math.ceil(per_sheet / columns))
    for start in range(0, len(rows), per_sheet):
        chunk = rows[start : start + per_sheet]
        figure, axes = plt.subplots(rows_per_sheet, columns, figsize=(4 * columns, 2.6 * rows_per_sheet))
        flat = np.atleast_1d(axes).ravel()
        for axis in flat:
            axis.axis("off")
        for axis, row in zip(flat, chunk):
            image_path = IMAGES_DIR / row["file_name"]
            if not image_path.exists():
                axis.set_title(f"{row['image_id']} MISSING", fontsize=8)
                continue
            with Image.open(image_path) as handle:
                axis.imshow(handle.convert("RGB"))
            axis.set_title(f"{row['image_id']}  {row['captured_at_utc'][:10]}", fontsize=8)
            axis.axis("off")
        figure.tight_layout()
        out_path = TRIAGE_DIR / f"sheet_{start // per_sheet + 1:02d}.jpg"
        figure.savefig(out_path, dpi=100, pil_kwargs={"quality": 85})
        plt.close(figure)
        print(f"  sheet {out_path.relative_to(REPO_ROOT)}  ({len(chunk)} images)")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in CSV_COLUMNS})


# main

def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def load_cache() -> dict[str, dict[str, Any]]:
    if not CACHE_FILE.exists():
        return {}
    try:
        payload = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[warn] ignoring unreadable cache {CACHE_FILE}: {exc}")
        return {}
    if not isinstance(payload, dict):
        return {}
    print(f"cache     : loaded {len(payload)} records from .cache/mapillary_records.json")
    return payload


def save_cache(records: dict[str, dict[str, Any]]) -> None:
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")


def run_fetch(args: argparse.Namespace) -> int:
    token = _token()
    session = _session()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)

    regions = _regions_from_args(args)
    allowed = {name for name, _ in regions}

    if args.force and not args.dry_run:
        cleared = 0
        for folder in (IMAGES_DIR, TRIAGE_DIR):
            if folder.exists():
                for path in folder.glob("*.jpg"):
                    path.unlink()
                    cleared += 1
        if cleared:
            print(f"cleared   : {cleared} stale images under data/images and data/triage")

    records: dict[str, dict[str, Any]] = {} if args.reset_cache else load_cache()
    status: dict[str, str] = {}
    tile_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    region_counts: dict[str, int] = defaultdict(int)
    requests_made = 0

    def urban_pool() -> dict[str, dict[str, Any]]:
        return {mid: rec for mid, rec in records.items() if rec.get("region") in allowed}

    if args.per_region:
        per_tile = args.per_region
    else:
        per_tile = max(50, math.ceil(args.candidates / 4))
    per_tile = min(per_tile, 2000)

    print(f"preset    : {args.preset}  ({len(regions)} region(s))")
    print(f"page size : {args.page_size} (auto-halves to {args.min_page_size} on HTTP 500)")
    print(f"per tile  : up to {per_tile} raw records, stop at {args.max_records} urban records")
    print(f"token     : {token[:12]}...  (redacted)")

    stop = False
    for region_name, region_bbox in regions:
        if stop:
            break
        region_bbox = tuple(float(value) for value in region_bbox)
        if args.cols and args.rows:
            cols, rows = args.cols, args.rows
        else:
            cols, rows = choose_grid(region_bbox, args.max_tile_area)
        tile_area = ((region_bbox[2] - region_bbox[0]) / cols) * ((region_bbox[3] - region_bbox[1]) / rows)
        if tile_area > MAX_API_BBOX_AREA:
            raise SystemExit(
                f"{region_name}: tile area {tile_area:.4f} deg^2 exceeds the API limit of "
                f"{MAX_API_BBOX_AREA}; lower --max-tile-area"
            )
        tiles = list(tile_bboxes(region_bbox, cols, rows))
        random.Random(args.shuffle_seed).shuffle(tiles)
        print(f"\n-- {region_name}: {cols}x{rows} = {len(tiles)} tile(s), {tile_area:.4f} deg^2/tile")

        for tile in tiles:
            if len(urban_pool()) >= args.max_records:
                print(f"  stop: {len(urban_pool())} urban records reached --max-records {args.max_records}")
                stop = True
                break

            page_size = args.page_size
            raw: list[dict[str, Any]] = []
            while True:
                try:
                    raw = search_tile(session, token, tile, per_tile, page_size, args.request_pause)
                    break
                except RequestFailed as exc:
                    message = str(exc)
                    if "reduce the amount of data" in message.lower() and page_size > args.min_page_size:
                        page_size = max(args.min_page_size, page_size // 2)
                        print(f"    retry with page_size={page_size}")
                        time.sleep(args.request_pause * 3)
                        continue
                    failures.append(
                        {"region": region_name,
                         "bbox": [round(value, 4) for value in tile],
                         "error": message}
                    )
                    print(f"    FAILED: {message[:140]}")
                    break

            requests_made += 1
            new_records = 0
            for record in raw:
                normalised = _normalise_record(record, args.size)
                if normalised is None:
                    continue
                if args.min_year and normalised["captured_ms"] is not None:
                    year = datetime.fromtimestamp(
                        normalised["captured_ms"] / 1000.0, tz=timezone.utc
                    ).year
                    if year < args.min_year:
                        continue
                mapillary_id = normalised["mapillary_id"]
                if mapillary_id in records:
                    continue
                normalised["region"] = region_name
                records[mapillary_id] = normalised
                region_counts[region_name] += 1
                new_records += 1

            tile_rows.append(
                {
                    "region": region_name,
                    "bbox": [round(value, 4) for value in tile],
                    "page_size": page_size,
                    "raw": len(raw),
                    "new": new_records,
                    "region_total": region_counts[region_name],
                }
            )
            print(f"    raw={len(raw):>4}  new={new_records:>4}  "
                  f"region_total={region_counts[region_name]:>4}  urban_pool={len(urban_pool())}")
            save_cache(records)
            time.sleep(args.request_pause)

    region_totals: dict[str, int] = defaultdict(int)
    for record in records.values():
        name = record.get("region")
        if name in allowed:
            region_totals[name] += 1

    print(f"\nurban records: {len(urban_pool())}")
    if failures:
        print(f"[warn] {len(failures)} tile(s) failed and were skipped")
    if len(urban_pool()) < args.target:
        print(f"[warn] only {len(urban_pool())} urban records; lower --page-size or rerun later")

    if args.dry_run:
        print("\n[dry-run] metadata only; nothing downloaded")
        print("\nper-region records:")
        for name, _ in regions:
            print(f"  {name:<28} {region_totals[name]:>4}")

    pool = urban_pool()
    by_sequence: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in sorted(pool.values(), key=lambda item: (item["captured_ms"] or 0, item["mapillary_id"])):
        by_sequence[record["sequence"]].append(record)

    groups = [
        spaced(group, args.max_per_sequence, args.min_gap_seconds)
        for group in by_sequence.values()
    ]
    shortlist = list(islice(round_robin(groups), args.candidates))
    print(f"regions used: {sum(1 for name in region_counts if region_counts[name])}")
    print(f"sequences   : {len(groups)}")
    print(f"shortlist   : {len(shortlist)} frames")

    accepted: list[dict[str, Any]] = []
    hashes: list[int] = []
    for position, record in enumerate(shortlist, 1):
        if len(accepted) >= args.target:
            break
        raw_path = RAW_DIR / f"{record['mapillary_id']}.jpg"
        if not raw_path.exists():
            try:
                download(session, record["thumb"], raw_path)
            except Exception as exc:
                status[record["mapillary_id"]] = f"download-failed: {exc}"
                continue
            requests_made += 1
            time.sleep(args.sleep)
        try:
            value = dhash(raw_path)
        except Exception as exc:
            status[record["mapillary_id"]] = f"unreadable: {exc}"
            continue
        if any(hamming(value, other) <= args.hamming for other in hashes):
            status[record["mapillary_id"]] = "near-duplicate"
            continue
        record["dhash"] = value
        accepted.append(record)
        hashes.append(value)
        status[record["mapillary_id"]] = "kept"
        if position % 25 == 0:
            print(f"  scanned {position}/{len(shortlist)}  accepted={len(accepted)}")

    for record in shortlist:
        status.setdefault(record["mapillary_id"], "not-scanned")

    if not accepted:
        raise SystemExit("no frames accepted - check the token, bbox and thumb URLs")

    print(f"accepted    : {len(accepted)} frames")
    if len(accepted) < args.target:
        print(f"[warn] {len(accepted)} < --target {args.target}; raise --candidates or --max-records")

    rows: list[dict[str, Any]] = []
    for index, record in enumerate(accepted, 1):
        raw_path = RAW_DIR / f"{record['mapillary_id']}.jpg"
        with Image.open(raw_path) as handle:
            original_width, original_height = handle.size
        file_name = f"{index:04d}.jpg"
        out_path = IMAGES_DIR / file_name
        width, height = normalise(raw_path, out_path, args.max_side)
        creator = record["creator"] or "Mapillary contributor"
        rows.append(
            {
                "image_id": index,
                "file_name": file_name,
                "mapillary_id": record["mapillary_id"],
                "captured_at_utc": record["captured_at_utc"],
                "lat": record["lat"],
                "lon": record["lon"],
                "region": record.get("region", ""),
                "sequence": record["sequence"],
                "compass_angle": record["compass_angle"],
                "creator": record["creator"],
                "license": LICENSE,
                "attribution": f"{creator} / Mapillary / {LICENSE}",
                "original_url": record["thumb"],
                "original_width": original_width,
                "original_height": original_height,
                "width": width,
                "height": height,
                "max_side": args.max_side,
                "sha256": sha256(out_path),
                "dhash": f"{record['dhash']:016x}",
                "keep": "yes",
            }
        )
        if index % 25 == 0:
            print(f"  normalised {index}/{len(accepted)}")

    write_csv(ANNOTATIONS_DIR / "attributes.csv", rows)

    pool_export = [
        {
            "mapillary_id": record["mapillary_id"],
            "region": record.get("region", ""),
            "sequence": record["sequence"],
            "captured_at_utc": record["captured_at_utc"],
            "lat": record["lat"],
            "lon": record["lon"],
            "status": status.get(record["mapillary_id"], "unknown"),
        }
        for record in records.values()
    ]
    (ANNOTATIONS_DIR / "candidate_pool.json").write_text(
        json.dumps(pool_export, indent=2), encoding="utf-8"
    )

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": _git_sha(),
        "purpose": "Phase 1 candidate pool for the Hong Kong street-scene test set",
        "endpoint": API_IMAGES,
        "preset": args.preset,
        "regions": {
            name: {
                "bbox": [round(float(value), 4) for value in bbox],
                "records": region_totals.get(name, 0),
            }
            for name, bbox in regions
        },
        "tiles": {"detail": tile_rows, "failed": failures},
        "license": LICENSE,
        "download_size": args.size,
        "max_side": args.max_side,
        "dedupe": {
            "max_per_sequence": args.max_per_sequence,
            "min_gap_seconds": args.min_gap_seconds,
            "dhash_hamming": args.hamming,
        },
        "counts": {
            "urban_records": len(pool),
            "regions_with_data": sum(1 for count in region_totals.values() if count),
            "sequences": len(groups),
            "shortlisted": len(shortlist),
            "accepted": len(accepted),
            "failed_tiles": len(failures),
            "api_requests": requests_made,
        },
        "files": {
            "attributes": "data/annotations/attributes.csv",
            "candidate_pool": "data/annotations/candidate_pool.json",
            "images": "data/images/<image_id>.jpg",
            "raw": "data/raw/<mapillary_id>.jpg",
        },
    }
    (ANNOTATIONS_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print("\noutputs:")
    print(f"  images      : data/images/0001.jpg ... {len(accepted):04d}.jpg")
    print("  attributes  : data/annotations/attributes.csv")
    print("  manifest    : data/annotations/manifest.json")
    print("  candidate   : data/annotations/candidate_pool.json")

    if args.sheets:
        print("\ncontact sheets:")
        write_sheets(rows, args.per_sheet, args.sheet_columns)
    return 0


def run_sheets(args: argparse.Namespace) -> int:
    csv_path = ANNOTATIONS_DIR / "attributes.csv"
    if not csv_path.exists():
        raise SystemExit(f"{csv_path} not found - run the fetch step first")
    rows = read_csv(csv_path)
    print(f"rendering {len(rows)} images into {TRIAGE_DIR}")
    write_sheets(rows, args.per_sheet, args.sheet_columns)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mapillary acquisition for the HK test set")
    parser.add_argument("--preset", choices=["urban", "all", "custom"], default="urban",
                        help="urban = 20 dense district boxes (default); all = whole HK; custom = --bbox")
    parser.add_argument("--bbox", nargs=4, type=float, default=None,
                        metavar=("WEST", "SOUTH", "EAST", "NORTH"))
    parser.add_argument("--per-region", type=int, default=50,
                        help="max raw records per region (0 = derive from --candidates)")
    parser.add_argument("--cols", type=int, default=0, help="bbox tiles horizontally (0 = auto)")
    parser.add_argument("--rows", type=int, default=0, help="bbox tiles vertically (0 = auto)")
    parser.add_argument("--max-tile-area", type=float, default=DEFAULT_TILE_AREA,
                        help="maximum square degrees per tile (API limit 0.010)")
    parser.add_argument("--page-size", type=int, default=100, help="records per API request")
    parser.add_argument("--min-page-size", type=int, default=25)
    parser.add_argument("--max-records", type=int, default=700,
                        help="stop after this many urban records (cache is reused)")
    parser.add_argument("--request-pause", type=float, default=0.4, help="seconds between API requests")
    parser.add_argument("--shuffle-seed", type=int, default=0, help="tile visiting order seed")
    parser.add_argument("--target", type=int, default=260, help="frames to keep")
    parser.add_argument("--candidates", type=int, default=600, help="frames to download and hash")
    parser.add_argument("--size", choices=["2048", "original", "1024"], default="2048",
                        help="which thumbnail to download")
    parser.add_argument("--max-side", type=int, default=1280,
                        help="long side of the normalised image; 0 keeps the original size")
    parser.add_argument("--max-per-sequence", type=int, default=2)
    parser.add_argument("--min-gap-seconds", type=int, default=10)
    parser.add_argument("--hamming", type=int, default=8, help="dHash distance treated as duplicate")
    parser.add_argument("--min-year", type=int, default=0, help="drop frames captured before this year")
    parser.add_argument("--sleep", type=float, default=0.2, help="seconds between downloads")
    parser.add_argument("--dry-run", action="store_true", help="metadata only, no download")
    parser.add_argument("--force", action="store_true", help="overwrite existing outputs")
    parser.add_argument("--sheets", action="store_true", help="render triage contact sheets")
    parser.add_argument("--sheets-only", action="store_true", help="only re-render contact sheets")
    parser.add_argument("--per-sheet", type=int, default=20)
    parser.add_argument("--sheet-columns", type=int, default=4)
    parser.add_argument("--reset-cache", action="store_true", help="ignore the local record cache")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.sheets_only:
        return run_sheets(args)

    # A dry run writes nothing, so it must not be blocked by the overwrite guard.
    if args.dry_run:
        return run_fetch(args)

    existing = [
        path
        for path in (ANNOTATIONS_DIR / "attributes.csv", ANNOTATIONS_DIR / "manifest.json")
        if path.exists()
    ]
    if existing and not args.force:
        raise SystemExit(
            "refusing to overwrite " + ", ".join(path.name for path in existing) +
            "; pass --force only if no annotation work has started yet"
        )
    return run_fetch(args)


if __name__ == "__main__":
    raise SystemExit(main())