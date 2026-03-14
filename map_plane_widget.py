#!/usr/bin/env python3
"""
Plane Tracker
-------------
Fetches a map snapshot with current aircraft positions and displays it.
Updates every 5 seconds with live plane data.

Usage:
    python map_plane_widget.py --lat 40.7128 --lon -74.0060 --zoom 7

Requirements:
    pip install requests pillow python-dotenv opencv-python numpy matplotlib
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import requests
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

MAP_STYLE  = "dark-v11"   # streets-v12 | dark-v11 | outdoors-v12
IMAGE_WIDTH = 800          # wide rectangle width in pixels
IMAGE_HEIGHT = 600         # wide rectangle height in pixels
SHADOW_OFFSET = 15         # pixels offset for shadow (higher = planes appear further up)
ALTITUDE_SCALE = 0.1      # pixels per meter of altitude (0.15 = 150 pixels per 1000m, ~1500 pixels at cruising altitude)
MAX_ALTITUDE = 5000       # maximum altitude in meters (cap for display, ~40,000 ft cruising)

TOKEN_URL = "https://auth.opensky-network.org/auth/realms/opensky-network/protocol/openid-connect/token"
TOKEN_REFRESH_MARGIN = 30  # seconds before expiry to refresh

MAP_REFETCH_THRESHOLD = 0.005  # degrees (~500 m at equator); pan beyond this to trigger a new tile fetch

DEFAULT_AIRLINE_COLORS = ((255, 255, 255), (200, 200, 200), (150, 150, 150))


_LIVERY_CACHE: dict | None = None


def _load_livery_data() -> dict:
    """Load livery data from livery.json once per process run."""
    global _LIVERY_CACHE
    if _LIVERY_CACHE is not None:
        return _LIVERY_CACHE

    livery_path = Path(__file__).with_name("livery.json")
    try:
        with livery_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            _LIVERY_CACHE = data if isinstance(data, dict) else {}
    except Exception:
        _LIVERY_CACHE = {}

    return _LIVERY_CACHE


class TokenManager:
    """Manages OpenSky OAuth2 access tokens with automatic refresh."""
    def __init__(self, client_id: str = None, client_secret: str = None):
        self.client_id = client_id
        self.client_secret = client_secret
        self.token = None
        self.expires_at = None

    def get_token(self):
        """Return a valid access token, refreshing automatically if needed."""
        if not self.client_id or not self.client_secret:
            return None
        
        if self.token and self.expires_at and datetime.now() < self.expires_at:
            return self.token
        
        return self._refresh()

    def _refresh(self):
        """Fetch a new access token from the OpenSky authentication server."""
        try:
            r = requests.post(
                TOKEN_URL,
                data={
                    "grant_type": "client_credentials",
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                },
                timeout=10
            )
            r.raise_for_status()

            data = r.json()
            self.token = data["access_token"]
            expires_in = data.get("expires_in", 1800)
            self.expires_at = datetime.now() + timedelta(seconds=expires_in - TOKEN_REFRESH_MARGIN)
            print(f"✓ OpenSky token obtained (expires in {expires_in}s)")
            return self.token
        except Exception as e:
            print(f"⚠️  Failed to get OpenSky token: {e}")
            return None

    def headers(self):
        """Return request headers with a valid Bearer token."""
        token = self.get_token()
        if token:
            return {"Authorization": f"Bearer {token}"}
        return {}


def enhance_map(img: Image.Image) -> Image.Image:
    """Apply futuristic styling to the map: only highlight major roads and borders."""
    img = img.convert("RGBA")
    arr = np.array(img, dtype=np.float32)   # shape (H, W, 4)
    r, g, b, a = arr[..., 0], arr[..., 1], arr[..., 2], arr[..., 3]

    brightness = (r + g + b) / 3
    bright = brightness > 50  # major roads / borders

    out = np.zeros_like(arr)
    # Bright pixels → cyan/blue glow
    out[bright, 0] = np.clip(r[bright] * 0.6,  0, 255)
    out[bright, 1] = np.clip(g[bright] * 1.5,  0, 255)
    out[bright, 2] = np.clip(b[bright] * 3.0,  0, 255)
    # Dark pixels → near-black background
    out[~bright, 0] = np.clip(r[~bright] * 0.05, 0, 255)
    out[~bright, 1] = np.clip(g[~bright] * 0.05, 0, 255)
    out[~bright, 2] = np.clip(b[~bright] * 0.05, 0, 255)
    out[..., 3] = a

    result = Image.fromarray(out.astype(np.uint8), "RGBA")
    return result.filter(ImageFilter.SHARPEN).convert("RGB")


def fetch_map(lat: float, lon: float, zoom: int, token: str, bearing: float = 0) -> Image.Image:
    url = (
        f"https://api.mapbox.com/styles/v1/mapbox/{MAP_STYLE}/static/"
        f"{lon},{lat},{zoom},{bearing}/"
        f"{IMAGE_WIDTH}x{IMAGE_HEIGHT}"
        f"?access_token={token}"
    )
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    base_map = Image.open(BytesIO(resp.content)).convert("RGB")
    return enhance_map(base_map)


def bounding_box(lat: float, lon: float, zoom: int) -> tuple:
    deg_per_px = 360.0 / (256 * 2 ** zoom)
    half_width = (IMAGE_WIDTH / 2) * deg_per_px
    half_height = (IMAGE_HEIGHT / 2) * deg_per_px
    return lat - half_height, lon - half_width, lat + half_height, lon + half_width


def fetch_planes(min_lat, min_lon, max_lat, max_lon, token_manager: TokenManager = None) -> list:
    params = dict(lamin=min_lat, lomin=min_lon, lamax=max_lat, lomax=max_lon)
    
    # Use token-based authentication if available
    headers = token_manager.headers() if token_manager else {}
    
    resp = requests.get(
        "https://opensky-network.org/api/states/all",
        params=params, headers=headers, timeout=15)
    resp.raise_for_status()
    return resp.json().get("states") or []


def geo_to_pixel(lon: float, lat: float,
                 center_lat: float, center_lon: float, zoom: int,
                 bearing: float = 0) -> tuple[int, int]:
    def merc_y(lat_deg):
        rad = math.radians(lat_deg)
        return math.log(math.tan(math.pi / 4 + rad / 2))

    scale = (256 * 2 ** zoom) / (2 * math.pi)
    cx = scale * (math.radians(center_lon) + math.pi)
    cy = scale * (math.pi - merc_y(center_lat))
    px = scale * (math.radians(lon) + math.pi)
    py = scale * (math.pi - merc_y(lat))
    dx, dy = px - cx, py - cy
    if bearing != 0:
        b = math.radians(bearing)
        cos_b, sin_b = math.cos(b), math.sin(b)
        dx, dy = dx * cos_b + dy * sin_b, -dx * sin_b + dy * cos_b
    return int(IMAGE_WIDTH / 2 + dx), int(IMAGE_HEIGHT / 2 + dy)


def get_airline_color(callsign: str) -> tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]:
    """Get color scheme (primary, secondary1, secondary2) based on airline from callsign prefix."""
    if not callsign:
        return DEFAULT_AIRLINE_COLORS

    livery = _load_livery_data()
    operators = livery.get("operators", {})
    brands = livery.get("brands", {})

    prefix = callsign[:3].upper()
    brand_key = operators.get(prefix) or operators.get(callsign[:2].upper())
    if not brand_key:
        return DEFAULT_AIRLINE_COLORS

    brand = brands.get(brand_key, {})
    colors = brand.get("colors", {})

    try:
        primary = tuple(colors.get("primary", DEFAULT_AIRLINE_COLORS[0]))
        secondary = tuple(colors.get("secondary", DEFAULT_AIRLINE_COLORS[1]))
        accent = tuple(colors.get("accent", DEFAULT_AIRLINE_COLORS[2]))
        if len(primary) == 3 and len(secondary) == 3 and len(accent) == 3:
            return primary, secondary, accent
    except Exception:
        pass

    return DEFAULT_AIRLINE_COLORS


def draw_connecting_line(draw: ImageDraw.ImageDraw, ground_pos: tuple[int, int], 
                        plane_pos: tuple[int, int], color: tuple[int, int, int]):
    """Draw a thin vertical line connecting ground to plane."""
    # Draw a thin line from ground to plane
    draw.line([ground_pos, plane_pos], fill=color, width=1)


def create_plane_image(heading: float | None, callsign: str, size: int = 140):
    """Create a plane image on a transparent canvas.
    
    Returns:
        PIL Image (RGBA) with the plane centered
    """
    # Create a transparent canvas
    img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Center position
    x, y = size // 2, size // 2
    
    angle = math.radians((heading or 0) - 90)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    
    # Plane scale factor
    scale = 2.5
    
    def rotate(dx, dy):
        """Rotate point (dx, dy) around origin by angle."""
        return (x + int(dx * scale * cos_a - dy * scale * sin_a),
                y + int(dx * scale * sin_a + dy * scale * cos_a))
    
    # Fuselage (main body)
    nose = rotate(10, 0)
    tail = rotate(-8, 0)
    
    # Wings (horizontal)
    wing_left = rotate(0, -8)
    wing_right = rotate(0, 8)
    wing_front = rotate(2, 0)
    
    # Tail fins (vertical stabilizer)
    tail_top = rotate(-8, -3)
    tail_bottom = rotate(-8, 3)
    
    # Get airline-specific 3-color scheme
    primary_color, secondary1_color, secondary2_color = get_airline_color(callsign)
    
    # Create darker version for glow
    glow_color = tuple(int(c * 0.3) for c in primary_color)
    
    # Draw glow effect (larger, semi-transparent)
    draw.line([wing_left, wing_front, wing_right], fill=glow_color, width=int(4 * scale))
    draw.line([nose, tail], fill=glow_color, width=int(5 * scale))
    
    # Create brighter version for fuselage
    bright_color = tuple(min(255, int(c * 1.2)) for c in primary_color)
    
    # Main wings with primary color
    draw.line([wing_left, wing_front, wing_right], fill=primary_color, width=int(2 * scale))
    # Fuselage - brighter primary
    draw.line([nose, tail], fill=bright_color, width=int(3 * scale))
    # Tail with primary color
    draw.line([tail_top, tail, tail_bottom], fill=primary_color, width=int(2 * scale))
    
    # Add secondary color accents on wingtips
    wingtip_size = int(1.2 * scale)
    draw.ellipse([wing_left[0]-wingtip_size, wing_left[1]-wingtip_size, 
                  wing_left[0]+wingtip_size, wing_left[1]+wingtip_size],
                 fill=secondary1_color)
    draw.ellipse([wing_right[0]-wingtip_size, wing_right[1]-wingtip_size, 
                  wing_right[0]+wingtip_size, wing_right[1]+wingtip_size],
                 fill=secondary1_color)
    
    # Add secondary2 color accent on tail
    draw.line([tail_top, tail_bottom], fill=secondary2_color, width=int(1.5 * scale))
    
    # Draw callsign label
    label = (callsign or "").strip()
    if label:
        draw.text((x + int(12 * scale), y - int(10 * scale)), label,
                  fill=primary_color, stroke_fill=(0, 0, 0), stroke_width=2)
    
    return img, primary_color


def warp_plane_at_position(plane_img: Image.Image, y_pos: float, 
                           image_height: int, top_shrink: float, vertical_shift: float):
    """Apply perspective warp to a plane based on its vertical position."""
    # Calculate the scale factor based on y position in the perspective
    # Planes higher up (lower y) should be smaller due to perspective
    
    # Normalize y position (0 at top, 1 at bottom)
    top_y = image_height * vertical_shift
    if y_pos < top_y:
        # Above the perspective top, use top scale
        scale_factor = top_shrink
    else:
        # Interpolate between top and bottom
        t = (y_pos - top_y) / (image_height - top_y)
        scale_factor = top_shrink + (1.0 - top_shrink) * t
    
    # Also apply horizontal squeeze based on position
    horizontal_scale = scale_factor
    
    # Resize the plane
    original_size = plane_img.size
    new_width = int(original_size[0] * horizontal_scale)
    new_height = int(original_size[1] * scale_factor)
    
    if new_width > 0 and new_height > 0:
        warped = plane_img.resize((new_width, new_height), Image.LANCZOS)
        return warped
    
    return plane_img


def draw_plane(draw: ImageDraw.ImageDraw, x: int, y: int,
               heading: float | None, callsign: str, altitude: float | None = None):
    """Legacy function - kept for compatibility. Use draw_plane_warped for perspective effect."""
    # Calculate altitude offset
    altitude_offset = 0
    if altitude is not None and altitude > 0:
        altitude_offset = int(altitude * ALTITUDE_SCALE)
    
    y = y - altitude_offset
    
    angle = math.radians((heading or 0) - 90)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    scale = 2.5
    
    def rotate(dx, dy):
        return (x + int(dx * scale * cos_a - dy * scale * sin_a),
                y + int(dx * scale * sin_a + dy * scale * cos_a))
    
    nose = rotate(10, 0)
    tail = rotate(-8, 0)
    wing_left = rotate(0, -8)
    wing_right = rotate(0, 8)
    wing_front = rotate(2, 0)
    tail_top = rotate(-8, -3)
    tail_bottom = rotate(-8, 3)
    
    primary_color, secondary1_color, secondary2_color = get_airline_color(callsign)
    glow_color = tuple(int(c * 0.3) for c in primary_color)
    
    draw.line([wing_left, wing_front, wing_right], fill=glow_color, width=int(4 * scale))
    draw.line([nose, tail], fill=glow_color, width=int(5 * scale))
    
    bright_color = tuple(min(255, int(c * 1.2)) for c in primary_color)
    
    draw.line([wing_left, wing_front, wing_right], fill=primary_color, width=int(2 * scale))
    draw.line([nose, tail], fill=bright_color, width=int(3 * scale))
    draw.line([tail_top, tail, tail_bottom], fill=primary_color, width=int(2 * scale))
    
    wingtip_size = int(1.2 * scale)
    draw.ellipse([wing_left[0]-wingtip_size, wing_left[1]-wingtip_size, 
                  wing_left[0]+wingtip_size, wing_left[1]+wingtip_size],
                 fill=secondary1_color)
    draw.ellipse([wing_right[0]-wingtip_size, wing_right[1]-wingtip_size, 
                  wing_right[0]+wingtip_size, wing_right[1]+wingtip_size],
                 fill=secondary1_color)
    
    draw.line([tail_top, tail_bottom], fill=secondary2_color, width=int(1.5 * scale))
    
    label = (callsign or "").strip()
    if label:
        draw.text((x + int(12 * scale), y - int(10 * scale)), label,
                  fill=primary_color, stroke_fill=(0, 0, 0), stroke_width=2)
    
    return primary_color


def draw_plane_warped(base_img: Image.Image, x: int, y: int,
                     heading: float | None, callsign: str, altitude: float | None,
                     image_height: int, top_shrink: float, vertical_shift: float):
    """Draw a plane with perspective warp applied based on its position.
    
    Args:
        base_img: The base image to draw on (will be modified)
        x, y: Ground position (before altitude adjustment)
        heading: Plane heading in degrees
        callsign: Plane callsign for coloring
        altitude: Altitude in meters
        image_height: Total image height for perspective calculation
        top_shrink: Perspective top shrink factor
        vertical_shift: Perspective vertical shift factor
    
    Returns:
        Primary color of the plane
    """
    # Calculate altitude offset
    altitude_offset = 0
    if altitude is not None and altitude > 0:
        altitude_offset = int(altitude * ALTITUDE_SCALE)
    
    plane_y = y - altitude_offset
    
    # Create the plane on a separate canvas
    plane_img, primary_color = create_plane_image(heading, callsign)
    
    # Apply perspective warp based on GROUND position (not elevated position)
    # This ensures warping is based on where the plane is on the map, not its altitude
    warped_plane = warp_plane_at_position(plane_img, y, image_height, 
                                          top_shrink, vertical_shift)
    
    # Calculate paste position (centered on plane location)
    paste_x = x - warped_plane.width // 2
    paste_y = plane_y - warped_plane.height // 2
    
    # Paste the warped plane onto the base image
    base_img.paste(warped_plane, (paste_x, paste_y), warped_plane)
    
    return primary_color
    
    # Return the plane's primary color for the connecting line
    return primary_color


def warp_to_trapezium(
    image,
    top_shrink,
    vertical_shift,
    bowl_strength: float = 0.5,
):
    """
    Warp an image into a curved trapezium.

    This first applies a standard perspective trapezium warp, then applies a
    paraboloid-style vertical displacement where the 4 corners remain fixed
    and the interior lifts toward the center.

    Parameters
    ----------
    image : numpy array (OpenCV image)
    top_shrink : float
        How narrow the top becomes (0.5-0.8 works well).
    vertical_shift : float
        How much the top edge moves downward.
    bowl_strength : float
        Strength of center lift. 0 gives a flat trapezium.

    Returns
    -------
    warped_image (RGBA with transparent background)
    """

    h, w = image.shape[:2]
    
    # Convert to RGBA if not already
    if image.shape[2] == 3:
        # Add alpha channel
        image = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)

    # original rectangle
    src = np.float32([
        [0, 0],
        [w, 0],
        [w, h],
        [0, h]
    ])

    # trapezium destination
    top_width = w * top_shrink
    margin = (w - top_width) / 2
    top_y = h * vertical_shift

    dst = np.float32([
        [margin, top_y],          # top left
        [w - margin, top_y],      # top right
        [w, h],                   # bottom right
        [0, h]                    # bottom left
    ])

    # compute perspective transform
    matrix = cv2.getPerspectiveTransform(src, dst)

    # Stage 1: flat trapezium warp with transparent background
    warped = cv2.warpPerspective(image, matrix, (w, h), 
                                  borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=(0, 0, 0, 0))

    # Stage 2: paraboloid lift with fixed corners.
    if bowl_strength <= 0:
        return warped

    x = np.linspace(-1.0, 1.0, w, dtype=np.float32)
    y_px = np.arange(h, dtype=np.float32)
    x_grid, y_px_grid = np.meshgrid(x, y_px)

    # Build curvature in trapezium-local vertical coordinates so the far top
    # edge (y = top_y) is exactly flat after warping.
    denom = max(1.0, float(h - 1 - top_y))
    y_panel = np.clip((y_px_grid - top_y) / denom, 0.0, 1.0)

    # 0 at left/right boundaries and at panel top boundary.
    # Bottom edge is intentionally allowed to curve to avoid extreme distortion.
    x_profile = np.clip(1.0 - x_grid * x_grid, 0.0, 1.0)
    y_profile = np.sqrt(y_panel)
    profile = x_profile * y_profile

    base_y = y_px_grid

    # remap() expects source coordinates. Positive displacement here produces
    # upward curvature in output (requested direction).
    # Do not clamp to the bottom row: that creates vertical streak artifacts.
    lift_px = bowl_strength * 0.22 * h * profile

    map_x = ((x_grid + 1.0) * 0.5 * (w - 1)).astype(np.float32)
    map_y = (base_y + lift_px).astype(np.float32)

    curved = cv2.remap(
        warped,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )

    return curved


def draw_compass(img: Image.Image, bearing: float) -> Image.Image:
    """Overlay a NESW compass rose at the bottom-right corner of img.

    The labels rotate with the map bearing so each directional panel shows
    the correct cardinal orientation.  The compass is drawn after warping so
    it is never perspective-distorted.
    """
    img = img.copy()
    draw = ImageDraw.Draw(img)

    margin = 15
    radius = 22
    cx = img.width  - margin - radius
    cy = img.height - margin - radius

    # Semi-transparent dark background circle for readability
    pad = 5
    draw.ellipse(
        [cx - radius - pad, cy - radius - pad,
         cx + radius + pad, cy + radius + pad],
        fill=(0, 0, 0, 160)
    )

    # Cardinals in clockwise order; index 0=N, 1=E, 2=S, 3=W
    cardinals = ['N', 'E', 'S', 'W']
    # Determine which cardinal is "up" on this map from its bearing
    up_idx = int(round(bearing / 90)) % 4

    # Screen unit vectors: up, right, down, left
    screen_dirs = [(0, -1), (1, 0), (0, 1), (-1, 0)]
    # Fine-tune text offset per screen direction so labels sit neatly outside
    text_nudge = [(0, -8), (6, -4), (0, 0), (-14, -4)]

    for i, (dx, dy) in enumerate(screen_dirs):
        label = cardinals[(up_idx + i) % 4]
        color = (255, 80, 80) if label == 'N' else (210, 210, 210)

        ax = int(cx + dx * radius)
        ay = int(cy + dy * radius)

        # Spoke line
        draw.line([(cx, cy), (ax, ay)], fill=color, width=2)

        # Arrowhead (small triangle at tip)
        tip_size = 4
        # Perpendicular unit vector for arrowhead width
        px, py = -dy, dx
        arrow = [
            (ax, ay),
            (ax - dx * tip_size + px * tip_size, ay - dy * tip_size + py * tip_size),
            (ax - dx * tip_size - px * tip_size, ay - dy * tip_size - py * tip_size),
        ]
        draw.polygon(arrow, fill=color)

        # Cardinal label
        nx, ny = text_nudge[i]
        draw.text((ax + nx, ay + ny), label, fill=color,
                  stroke_fill=(0, 0, 0), stroke_width=2)

    # Centre dot
    dot_r = 3
    draw.ellipse([cx - dot_r, cy - dot_r, cx + dot_r, cy + dot_r],
                 fill=(255, 220, 0))

    return img


def create_hologram_cross(img_south: Image.Image, img_north: Image.Image,
                          img_east: Image.Image, img_west: Image.Image) -> Image.Image:
    """
    Create a hologram cross layout using four directional map views.

    Each panel uses a map rendered from the matching cardinal bearing so that
    the geometry and plane headings are correct for every face of the pyramid.

    Layout:
           [Top   - north view, bearing 180°]
    [Left  - east view, bearing 270°]  [Right - west view, bearing 90°]
           [Bottom - south view, bearing   0°]
    """
    # Ensure all inputs are RGBA
    def to_rgba(img):
        return img if img.mode == 'RGBA' else img.convert('RGBA')

    img_south = to_rgba(img_south)
    img_north = to_rgba(img_north)
    img_east  = to_rgba(img_east)
    img_west  = to_rgba(img_west)

    width, height = img_south.size

    # Apply hologram display transforms (same geometry as before)
    left  = img_east.rotate(90,  expand=True, fillcolor=(0, 0, 0, 0)).transpose(Image.FLIP_TOP_BOTTOM)
    right = img_west.rotate(-90, expand=True, fillcolor=(0, 0, 0, 0)).transpose(Image.FLIP_TOP_BOTTOM)

    # Minimal gap between images
    gap_x = 50
    gap_y = 570

    canvas_width  = left.width + gap_x + width + gap_x + right.width
    vertical_for_topbottom = height + gap_y + height
    vertical_for_sides     = max(left.height, right.height)
    canvas_height = max(vertical_for_topbottom, vertical_for_sides)

    canvas   = Image.new('RGBA', (canvas_width, canvas_height), (0, 0, 0, 255))
    center_x = left.width + gap_x

    # Bottom: south view (bearing 0°), rotated 180° then flipped left-right
    bottom   = img_south.rotate(180, expand=False, fillcolor=(0, 0, 0, 0)).transpose(Image.FLIP_LEFT_RIGHT)
    bottom_y = canvas_height - height
    canvas.paste(bottom, (center_x, bottom_y), bottom)

    # Top: north view (bearing 180°), flipped left-right
    top = img_north.copy().transpose(Image.FLIP_LEFT_RIGHT)
    canvas.paste(top, (center_x, 0), top)

    # Left: east view (bearing 270°), rotated 90° then flipped top-bottom
    left_y = (canvas_height - left.height) // 2
    canvas.paste(left, (0, left_y), left)

    # Right: west view (bearing 90°), rotated -90° then flipped top-bottom
    right_y = (canvas_height - right.height) // 2
    right_x = canvas_width - right.width
    canvas.paste(right, (right_x, right_y), right)

    # Composite onto black for final RGB output
    final = Image.new('RGB', (canvas_width, canvas_height), (0, 0, 0))
    final.paste(canvas, (0, 0), canvas)
    return final


def composite(base_map: Image.Image, planes: list,
              lat: float, lon: float, zoom: int, bearing: float = 0) -> Image.Image:
    # Use fixed vertical expansion based on MAX_ALTITUDE to keep canvas size constant
    vertical_expansion = int(MAX_ALTITUDE * ALTITUDE_SCALE) + 100  # Extra padding
    expanded_height = IMAGE_HEIGHT + vertical_expansion
    
    # Create expanded canvas with map at the bottom
    expanded_img = Image.new('RGB', (IMAGE_WIDTH, expanded_height), (0, 0, 0))
    expanded_img.paste(base_map, (0, vertical_expansion))
    
    # Apply slight blur to map for depth of field effect
    expanded_img = expanded_img.filter(ImageFilter.GaussianBlur(radius=0.5))
    
    # Convert to numpy and warp ONLY the map to trapezoid
    img_array = np.array(expanded_img)
    
    # Apply trapezoid/curvature transformation parameters
    top_shrink = 0.3
    vertical_shift = 0.8
    bowl_strength = 0.45
    
    h, w = img_array.shape[:2]
    
    # Convert to RGBA if not already
    if img_array.shape[2] == 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2RGBA)
    
    # Define the perspective transform
    src = np.float32([
        [0, 0],
        [w, 0],
        [w, h],
        [0, h]
    ])
    
    top_width = w * top_shrink
    margin = (w - top_width) / 2
    top_y = h * vertical_shift
    
    dst = np.float32([
        [margin, top_y],          # top left
        [w - margin, top_y],      # top right
        [w, h],                   # bottom right
        [0, h]                    # bottom left
    ])
    
    # Get the perspective transformation matrix
    matrix = cv2.getPerspectiveTransform(src, dst)
    
    # Warp the map using the same curved trapezium function used elsewhere.
    warped_map_array = warp_to_trapezium(
        img_array,
        top_shrink=top_shrink,
        vertical_shift=vertical_shift,
        bowl_strength=bowl_strength,
    )
    
    # Convert warped map back to PIL Image
    warped_img = Image.fromarray(warped_map_array, 'RGBA')
    draw = ImageDraw.Draw(warped_img)
    alpha_mask = np.array(warped_img)[..., 3]

    def is_on_map(px: int, py: int) -> bool:
        """True when a pixel lies on non-transparent warped map content."""
        if px < 0 or py < 0 or px >= w or py >= h:
            return False
        return alpha_mask[py, px] > 10
    
    # Function to transform coordinates using only trapezium perspective.
    # The bowl warp is applied to the map texture only.
    def transform_point(x, y):
        """Transform a point through perspective only."""
        pt = np.array([[[x, y]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(pt, matrix)
        px = float(transformed[0][0][0])
        py = float(transformed[0][0][1])

        return int(px), int(py)
    
    # Store plane data for two-pass rendering (no shadows)
    plane_data = []
    
    # First pass: collect plane positions (adjusted for expanded canvas)
    for state in planes:
        try:
            plon, plat = state[5], state[6]
            altitude = state[7]  # barometric altitude in meters
            if plon is None or plat is None:
                continue
            # Cap altitude at MAX_ALTITUDE
            if altitude is not None and altitude > MAX_ALTITUDE:
                altitude = MAX_ALTITUDE
            x, y = geo_to_pixel(plon, plat, lat, lon, zoom, bearing)
            # Adjust y for expanded canvas
            y_adjusted = y + vertical_expansion
            if 0 <= x < IMAGE_WIDTH and 0 <= y < IMAGE_HEIGHT:
                # Transform the ground position through perspective
                ground_x, ground_y = transform_point(x, y_adjusted)
                if not is_on_map(ground_x, ground_y):
                    continue
                # Adjust heading so the icon faces correctly on the rotated map
                raw_heading = state[10]
                adjusted_heading = ((raw_heading - bearing) % 360) if raw_heading is not None else None
                plane_data.append({
                    'ground_pos': (ground_x, ground_y),
                    'heading': adjusted_heading,
                    'callsign': state[1] or "",
                    'altitude': altitude
                })
        except (IndexError, TypeError):
            continue
    
    # Second pass: draw connecting lines
    for data in plane_data:
        altitude = data['altitude']
        if altitude is not None and altitude > 0:
            # Calculate plane position with altitude offset (upward from transformed ground position)
            altitude_offset = int(altitude * ALTITUDE_SCALE)
            plane_pos = (data['ground_pos'][0], data['ground_pos'][1] - altitude_offset)
            
            # Get the plane's primary color for the line
            primary_color, _, _ = get_airline_color(data['callsign'])
            
            # Draw connecting line from ground to plane
            draw_connecting_line(draw, data['ground_pos'], plane_pos, primary_color)
    
    # Third pass: draw all planes on top with perspective warp applied
    for data in plane_data:
        x, y = data['ground_pos']
        plane_color = draw_plane_warped(warped_img, x, y, data['heading'], data['callsign'], 
                                       data['altitude'], h, top_shrink, vertical_shift)
    
    # Find the bounding box of actual content (non-black pixels)
    # Convert to numpy for analysis
    # Use fixed crop dimensions to prevent size changes when planes move
    # Crop from bottom of image (where map is) upward by a fixed height
    fixed_display_height = 700  # Fixed height in pixels
    bottom_crop = warped_img.height
    top_crop = max(0, bottom_crop - fixed_display_height)
    
    # Crop the image to fixed dimensions
    warped_img = warped_img.crop((0, top_crop, warped_img.width, bottom_crop))
    
    # Scale down the image to make the 4 images in the cross smaller
    scale_factor = 0.6  # Scale to 60% of original size
    new_width = int(warped_img.width * scale_factor)
    new_height = int(warped_img.height * scale_factor)
    warped_img = warped_img.resize((new_width, new_height), Image.LANCZOS)

    # Overlay compass rose (not warped)
    warped_img = draw_compass(warped_img, bearing)

    return warped_img


def run(lat: float, lon: float, zoom: int, joystick=None):
    """Main display loop.

    Args:
        lat, lon: Initial map centre coordinates.
        zoom:     Mapbox zoom level.
        joystick: Optional Joystick instance.  If None, the map stays fixed.
    """
    load_dotenv()
    token = os.getenv("MAPBOX_TOKEN")
    token_manager = TokenManager(os.getenv("clientId"), os.getenv("clientSecret"))

    plt.ion()
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111)
    ax.axis('off')
    fig.patch.set_facecolor('black')
    fig.tight_layout(pad=0)

    mng = plt.get_current_fig_manager()
    try:
        mng.frame.Maximize(True)
    except Exception:
        try:
            mng.window.showMaximized()
        except Exception:
            pass

    plt.show()

    POLL_HZ = 10          # joystick poll rate (Hz)
    PLANE_INTERVAL = 5.0  # seconds between OpenSky fetches

    maps = None
    last_map_lat = last_map_lon = None
    img_display = None

    try:
        while True:
            # Re-fetch map tiles when the centre has moved beyond the threshold (or on first run)
            if maps is None or (
                abs(lat - last_map_lat) > MAP_REFETCH_THRESHOLD
                or abs(lon - last_map_lon) > MAP_REFETCH_THRESHOLD
            ):
                print(f"Fetching maps for ({lat:.4f}, {lon:.4f})…")
                maps = {
                    0:   fetch_map(lat, lon, zoom, token, bearing=0),
                    180: fetch_map(lat, lon, zoom, token, bearing=180),
                    270: fetch_map(lat, lon, zoom, token, bearing=270),
                    90:  fetch_map(lat, lon, zoom, token, bearing=90),
                }
                last_map_lat, last_map_lon = lat, lon

            print("Fetching planes…")
            planes = fetch_planes(*bounding_box(lat, lon, zoom), token_manager)
            print(f"{len(planes)} aircraft found: {', '.join([p[1] or 'N/A' for p in planes[:60]])}")

            comp_south = composite(maps[0],   planes, lat, lon, zoom, bearing=0)
            comp_north = composite(maps[180],  planes, lat, lon, zoom, bearing=180)
            comp_east  = composite(maps[270],  planes, lat, lon, zoom, bearing=270)
            comp_west  = composite(maps[90],   planes, lat, lon, zoom, bearing=90)
            result = create_hologram_cross(comp_south, comp_north, comp_east, comp_west)

            if img_display is None:
                img_display = ax.imshow(np.array(result))
            else:
                img_display.set_data(np.array(result))
            plt.draw()

            # Poll joystick at POLL_HZ while waiting for the next plane fetch
            ticks = int(PLANE_INTERVAL * POLL_HZ)
            for _ in range(ticks):
                if joystick is not None:
                    print("JOYSTICK CHANGE:")
                    dlon, dlat = joystick.get_pan_delta()
                    print(dlon, dlat)
                    lat  += dlat
                    lon  += dlon
                plt.pause(1.0 / POLL_HZ)

    except KeyboardInterrupt:
        print("\nShutting down...")
        plt.close()


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Plane tracker")
    parser.add_argument("--lat",  type=float, default=41.978611, help="Center latitude")
    parser.add_argument("--lon",  type=float, default=-87.904724, help="Center longitude")
    parser.add_argument("--zoom", type=int,   default=11,        help="Zoom level (1-12)")
    args = parser.parse_args()
    run(lat=args.lat, lon=args.lon, zoom=args.zoom)


if __name__ == "__main__":
    main()
