#!/usr/bin/env python3
"""
Live Plane Tracker
------------------
Displays a live map with real-time aircraft positions.

Usage:
    python map_plane_widget.py --lat 40.7128 --lon -74.0060 --zoom 7

Requirements:
    pip install requests pillow python-dotenv
"""
from __future__ import annotations

import argparse
import math
import threading
import time
import tkinter as tk
from io import BytesIO

import requests
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageTk
import os


class PlaneTracker:

    MAP_STYLE    = "dark-v11"   # streets-v12 | dark-v11 | outdoors-v12
    IMAGE_SIZE   = 800          # square pixels
    REFRESH_SECS = 10          # OpenSky poll interval

    def __init__(self, lat: float, lon: float, zoom: int):
        load_dotenv()
        self.mapbox_token = os.getenv("MAPBOX_TOKEN")

        self.lat   = lat
        self.lon   = lon
        self.zoom  = zoom

        self.base_map = None
        self.planes   = []
        self._stop    = threading.Event()
        self._timer   = None

        self.root = tk.Tk()
        self.root.title(f"Live Plane Tracker  —  {lat:.4f}, {lon:.4f}")
        self.root.resizable(False, False)

        self.label = tk.Label(self.root, bg="black")
        self.label.pack()

        self.status_var = tk.StringVar(value="Fetching map…")
        tk.Label(self.root, textvariable=self.status_var,
                 bg="#1a1a1a", fg="#aaaaaa", pady=4).pack(fill="x")

        self.tk_img = None
        self.root.protocol("WM_DELETE_WINDOW", self.stop)

    # ── Public ────────────────────────────────────────────────────────────────

    def run(self):
        """Start the app — fetches map then enters the GUI main loop."""
        threading.Thread(target=self._init_map, daemon=True).start()
        self.root.mainloop()

    def stop(self):
        self._stop.set()
        if self._timer:
            self._timer.cancel()
        self.root.destroy()
    
    # Update the graphic with new latitude and longitude! 
    def recenter(self, lat: float, lon: float, zoom):
        self.lat = lat
        self.lon = lon
        if zoom is not None:
            self.zoom = zoom
        self.root.title(f"{lat:.4f}, {lon:.4f}")
        if self._timer:
            self._timer.cancel()
        self._stop.clear()
        threading.Thread(target=self._init_map, daemon=True).start()

    # ── Map fetching ──────────────────────────────────────────────────────────

    def _fetch_map(self) -> Image.Image:
        url = (
            f"https://api.mapbox.com/styles/v1/mapbox/{self.MAP_STYLE}/static/"
            f"{self.lon},{self.lat},{self.zoom},0/"
            f"{self.IMAGE_SIZE}x{self.IMAGE_SIZE}"
            f"?access_token={self.mapbox_token}"
        )
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        return Image.open(BytesIO(resp.content)).convert("RGB")

    def _init_map(self):
        self.status_var.set("Fetching base map…")
        try:
            self.base_map = self._fetch_map()
            self.status_var.set("Base map loaded. Fetching planes…")
            self._refresh_planes()
        except Exception as e:
            self.status_var.set(f"Map error: {e}")

    # ── Plane fetching ────────────────────────────────────────────────────────

    def _bounding_box(self):
        """Return the geographic bounding box visible in the image."""
        deg_per_px = 360.0 / (256 * 2 ** self.zoom)
        half = (self.IMAGE_SIZE / 2) * deg_per_px
        return (
            self.lat - half,   # min_lat
            self.lon - half,   # min_lon
            self.lat + half,   # max_lat
            self.lon + half,   # max_lon
        )

    def _fetch_planes(self, min_lat, min_lon, max_lat, max_lon) -> list:
        params = dict(lamin=min_lat, lomin=min_lon, lamax=max_lat, lomax=max_lon)
        try:
            resp = requests.get(
                "https://opensky-network.org/api/states/all",
                params=params, timeout=15)
            resp.raise_for_status()
            return resp.json().get("states") or []
        except Exception as e:
            print(f"[OpenSky] Error: {e}")
            return []

    def _refresh_planes(self):
        if self._stop.is_set():
            return
        bbox = self._bounding_box()
        self.planes = self._fetch_planes(*bbox)
        self._render()
        count = len(self.planes)
        self.status_var.set(
            f"{count} aircraft visible  ·  next update in {self.REFRESH_SECS}s")
        self._timer = threading.Timer(self.REFRESH_SECS, self._refresh_planes)
        self._timer.daemon = True
        self._timer.start()

    # ── Rendering ─────────────────────────────────────────────────────────────

    def _geo_to_pixel(self, lon: float, lat: float) -> tuple[int, int]:
        def merc_y(lat_deg):
            rad = math.radians(lat_deg)
            return math.log(math.tan(math.pi / 4 + rad / 2))

        scale = (256 * 2 ** self.zoom) / (2 * math.pi)
        cx = scale * (math.radians(self.lon) + math.pi)
        cy = scale * (math.pi - merc_y(self.lat))
        px = scale * (math.radians(lon) + math.pi)
        py = scale * (math.pi - merc_y(lat))
        x = int(self.IMAGE_SIZE / 2 + (px - cx))
        y = int(self.IMAGE_SIZE / 2 + (py - cy))
        return x, y

    def _draw_plane(self, draw: ImageDraw.ImageDraw,
                    x: int, y: int,
                    heading: float | None,
                    callsign: str):
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

    def _composite(self) -> Image.Image:
        img  = self.base_map.copy()
        draw = ImageDraw.Draw(img)
        size = self.IMAGE_SIZE

        for state in self.planes:
            try:
                lon, lat = state[5], state[6]
                if lon is None or lat is None:
                    continue
                heading  = state[10]
                callsign = state[1] or ""
                x, y = self._geo_to_pixel(lon, lat)
                if 0 <= x < size and 0 <= y < size:
                    self._draw_plane(draw, x, y, heading, callsign)
            except (IndexError, TypeError):
                continue

        ts = time.strftime("%H:%M:%S UTC", time.gmtime())
        draw.text((8, 8), f"✈  {len(self.planes)} aircraft  ·  {ts}",
                  fill=(255, 255, 255), stroke_fill=(0, 0, 0), stroke_width=2)
        return img

    def _render(self):
        if self.base_map is None:
            return
        img = self._composite()
        self.tk_img = ImageTk.PhotoImage(img)
        self.label.config(image=self.tk_img)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Live plane tracker")
    parser.add_argument("--lat",  type=float, default=40.7128, help="Center latitude")
    parser.add_argument("--lon",  type=float, default=-74.0060, help="Center longitude")
    parser.add_argument("--zoom", type=int,   default=7,        help="Zoom level (1-12)")
    args = parser.parse_args()

    PlaneTracker(args.lat, args.lon, args.zoom).run()


if __name__ == "__main__":
    main()