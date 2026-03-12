#!/usr/bin/env python3
"""
Plane Tracker
-------------
Fetches a map snapshot with current aircraft positions and displays it.

Usage:
    python map_plane_widget.py --lat 40.7128 --lon -74.0060 --zoom 7

Requirements:
    pip install requests pillow python-dotenv
"""
from __future__ import annotations

import argparse
import math
import os
import time
from io import BytesIO

import requests
from dotenv import load_dotenv
from PIL import Image, ImageDraw


MAP_STYLE  = "dark-v11"   # streets-v12 | dark-v11 | outdoors-v12
IMAGE_SIZE = 500           # square pixels


def fetch_map(lat: float, lon: float, zoom: int, token: str) -> Image.Image:
    url = (
        f"https://api.mapbox.com/styles/v1/mapbox/{MAP_STYLE}/static/"
        f"{lon},{lat},{zoom},0/"
        f"{IMAGE_SIZE}x{IMAGE_SIZE}"
        f"?access_token={token}"
    )
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    return Image.open(BytesIO(resp.content)).convert("RGB")


def bounding_box(lat: float, lon: float, zoom: int) -> tuple:
    deg_per_px = 360.0 / (256 * 2 ** zoom)
    half = (IMAGE_SIZE / 2) * deg_per_px
    return lat - half, lon - half, lat + half, lon + half


def fetch_planes(min_lat, min_lon, max_lat, max_lon) -> list:
    params = dict(lamin=min_lat, lomin=min_lon, lamax=max_lat, lomax=max_lon)
    resp = requests.get(
        "https://opensky-network.org/api/states/all",
        params=params, timeout=15)
    resp.raise_for_status()
    return resp.json().get("states") or []


def geo_to_pixel(lon: float, lat: float,
                 center_lat: float, center_lon: float, zoom: int) -> tuple[int, int]:
    def merc_y(lat_deg):
        rad = math.radians(lat_deg)
        return math.log(math.tan(math.pi / 4 + rad / 2))

    scale = (256 * 2 ** zoom) / (2 * math.pi)
    cx = scale * (math.radians(center_lon) + math.pi)
    cy = scale * (math.pi - merc_y(center_lat))
    px = scale * (math.radians(lon) + math.pi)
    py = scale * (math.pi - merc_y(lat))
    return int(IMAGE_SIZE / 2 + (px - cx)), int(IMAGE_SIZE / 2 + (py - cy))


def draw_plane(draw: ImageDraw.ImageDraw, x: int, y: int,
               heading: float | None, callsign: str):
    r = 6
    draw.ellipse([x - r, y - r, x + r, y + r],
                 fill=(255, 220, 0), outline=(255, 140, 0), width=2)
    if heading is not None:
        angle = math.radians(heading - 90)
        tip_x = x + int(math.cos(angle) * (r + 8))
        tip_y = y + int(math.sin(angle) * (r + 8))
        draw.line([x, y, tip_x, tip_y], fill=(255, 220, 0), width=2)
    label = (callsign or "").strip()
    if label:
        draw.text((x + r + 3, y - 6), label,
                  fill=(200, 200, 200), stroke_fill=(0, 0, 0), stroke_width=1)


def composite(base_map: Image.Image, planes: list,
              lat: float, lon: float, zoom: int) -> Image.Image:
    img  = base_map.copy()
    draw = ImageDraw.Draw(img)

    for state in planes:
        try:
            plon, plat = state[5], state[6]
            if plon is None or plat is None:
                continue
            x, y = geo_to_pixel(plon, plat, lat, lon, zoom)
            if 0 <= x < IMAGE_SIZE and 0 <= y < IMAGE_SIZE:
                draw_plane(draw, x, y, state[10], state[1] or "")
        except (IndexError, TypeError):
            continue

    ts = time.strftime("%H:%M:%S UTC", time.gmtime())
    draw.text((8, 8), f"✈  {len(planes)} aircraft  ·  {ts}",
              fill=(255, 255, 255), stroke_fill=(0, 0, 0), stroke_width=2)
    return img


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Plane tracker")
    parser.add_argument("--lat",  type=float, default=40.7128, help="Center latitude")
    parser.add_argument("--lon",  type=float, default=-74.0060, help="Center longitude")
    parser.add_argument("--zoom", type=int,   default=7,        help="Zoom level (1-12)")
    args = parser.parse_args()

    token = os.getenv("MAPBOX_TOKEN")
    print("Fetching map…")
    base_map = fetch_map(args.lat, args.lon, args.zoom, token)
    print("Fetching planes…")
    planes = fetch_planes(*bounding_box(args.lat, args.lon, args.zoom))
    print(f"{len(planes)} aircraft found.")
    composite(base_map, planes, args.lat, args.lon, args.zoom).show()


if __name__ == "__main__":
    main()
