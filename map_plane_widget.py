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
import math
import os
import time
from io import BytesIO

import cv2
import numpy as np
import requests
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

MAP_STYLE  = "dark-v11"   # streets-v12 | dark-v11 | outdoors-v12
IMAGE_WIDTH = 800          # wide rectangle width in pixels
IMAGE_HEIGHT = 500         # wide rectangle height in pixels
SHADOW_OFFSET = 15         # pixels offset for shadow (higher = planes appear further up)
ALTITUDE_SCALE = 0.15      # pixels per meter of altitude (0.15 = 150 pixels per 1000m, ~1500 pixels at cruising altitude)
MAX_ALTITUDE = 8000       # maximum altitude in meters (cap for display, ~40,000 ft cruising)

TOKEN_URL = "https://auth.opensky-network.org/auth/realms/opensky-network/protocol/openid-connect/token"
TOKEN_REFRESH_MARGIN = 30  # seconds before expiry to refresh


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
    # Convert to RGBA for processing
    img = img.convert("RGBA")
    pixels = img.load()
    width, height = img.size
    
    # Process each pixel to create minimal futuristic look
    for y in range(height):
        for x in range(width):
            r, g, b, a = pixels[x, y]
            
            # Calculate brightness to determine feature importance
            brightness = (r + g + b) / 3
            
            # Only highlight MAJOR roads and borders (very bright pixels)
            if brightness > 50:  # Major roads and borders only
                # Make major roads glow cyan/blue
                contrast = 3.0
                r = int(r * contrast * 0.2)  # Heavy reduce red
                g = int(g * contrast * 0.5)  # Some green  
                b = int(b * contrast * 1.0)  # Strong boost blue
            else:  # Background - pure black
                r = int(r * 0.05)
                g = int(g * 0.05)
                b = int(b * 0.05)
            
            # Clamp values
            pixels[x, y] = (min(255, max(0, r)), 
                           min(255, max(0, g)), 
                           min(255, max(0, b)), a)
    
    # Apply slight sharpening to make major roads crisp
    img = img.filter(ImageFilter.SHARPEN)
    
    return img.convert("RGB")


def fetch_map(lat: float, lon: float, zoom: int, token: str) -> Image.Image:
    url = (
        f"https://api.mapbox.com/styles/v1/mapbox/{MAP_STYLE}/static/"
        f"{lon},{lat},{zoom},0/"
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
                 center_lat: float, center_lon: float, zoom: int) -> tuple[int, int]:
    def merc_y(lat_deg):
        rad = math.radians(lat_deg)
        return math.log(math.tan(math.pi / 4 + rad / 2))

    scale = (256 * 2 ** zoom) / (2 * math.pi)
    cx = scale * (math.radians(center_lon) + math.pi)
    cy = scale * (math.pi - merc_y(center_lat))
    px = scale * (math.radians(lon) + math.pi)
    py = scale * (math.pi - merc_y(lat))
    return int(IMAGE_WIDTH / 2 + (px - cx)), int(IMAGE_HEIGHT / 2 + (py - cy))


def get_airline_color(callsign: str) -> tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]:
    """Get color scheme (primary, secondary1, secondary2) based on airline from callsign prefix."""
    if not callsign:
        return ((255, 255, 255), (200, 200, 200), (150, 150, 150))  # White for unknown
    
    # Extract airline code (usually first 3 characters)
    prefix = callsign[:3].upper()
    
    # Airline color mapping (ICAO code prefix -> (primary, secondary1, secondary2))
    airline_colors = {
        # US Airlines
        'AAL': ((0, 53, 149), (190, 10, 48), (167, 169, 172)),      # American - Blue, Red, Shadow Gray
        'UAL': ((0, 34, 101), (0, 92, 184), (107, 117, 126)),      # United - Blue Sphere, Rhapsody Blue, Sky Gray
        'DAL': ((224, 0, 52), (0, 50, 100), (184, 185, 188)),      # Delta - Red, Dark Blue, Silver
        'SWA': ((48, 71, 142), (245, 184, 0), (180, 49, 35)),       # Southwest - Bold Blue, Sunrise Yellow, Warm Red
        'JBU': ((0, 32, 91), (0, 156, 222), (255, 255, 255)),      # JetBlue - Midnight Blue, Bluebird, White
        'ASA': ((0, 66, 95), (0, 133, 161), (129, 204, 219)),      # Alaska - Midnight Blue, Tropical Blue, Breezy Blue
        'FFT': ((0, 102, 71), (173, 210, 54), (50, 50, 50)),       # Frontier - Frontier Green, Lime, Dark Gray
        'NKS': ((255, 236, 0), (0, 0, 0), (255, 255, 255)),        # Spirit - Yellow, Black, White
        
        # European Airlines
        'BAW': ((7, 34, 103), (235, 30, 35), (255, 255, 255)),     # British Airways - Blue, Red, White
        'RYR': ((0, 51, 153), (241, 194, 0), (255, 255, 255)),     # Ryanair - Blue, Yellow, White
        'EZY': ((255, 102, 0), (255, 255, 255), (51, 51, 51)),     # EasyJet - Orange, White, Dark Grey
        'DLH': ((0, 47, 95), (247, 168, 27), (255, 255, 255)),     # Lufthansa - Navy Blue, Deep Yellow, White
        'AFR': ((0, 35, 119), (237, 27, 46), (255, 255, 255)),     # Air France - Blue, Red, White
        'KLM': ((0, 161, 222), (0, 49, 117), (255, 255, 255)),     # KLM - Sky Blue, Dark Blue, White
        'IBE': ((181, 16, 26), (250, 175, 5), (255, 255, 255)),    # Iberia - Red, Yellow, White
        'AZA': ((0, 104, 71), (196, 18, 48), (255, 255, 255)),     # Alitalia - Green, Red, White
        
        # Asian Airlines
        'ANA': ((0, 49, 134), (0, 160, 223), (255, 255, 255)),     # ANA - Tritone Blue, Light Blue, White
        'JAL': ((217, 0, 19), (0, 0, 0), (255, 255, 255)),         # Japan Airlines - Red, Black, White
        'CPA': ((0, 101, 114), (255, 255, 255), (163, 171, 175)),  # Cathay Pacific - Brushwing Green, White, Gray
        'SIA': ((0, 51, 102), (255, 165, 0), (255, 255, 255)),     # Singapore - Dark Blue, Gold, White
        'THA': ((78, 35, 126), (241, 181, 21), (189, 0, 119)),     # Thai - Royal Purple, Gold, Magenta
        'QTR': ((141, 20, 59), (91, 94, 98), (255, 255, 255)),     # Qatar - Maroon, Gray, White
        'UAE': ((186, 151, 93), (255, 0, 0), (0, 0, 0)),           # Emirates - Gold, Red, Black
        'ETD': ((197, 160, 93), (35, 31, 32), (255, 255, 255)),    # Etihad - Desert Gold, Carbon, White
        
        # Other major airlines
        'QFA': ((228, 0, 43), (0, 0, 0), (255, 255, 255)),         # Qantas - Red, Black, White
        'ACA': ((0, 0, 0), (216, 12, 45), (255, 255, 255)),        # Air Canada - Black, Red, White
        'AMX': ((13, 35, 64), (220, 24, 45), (167, 169, 172)),     # Aeromexico - Dark Blue, Red, Silver
        'TAM': ((15, 30, 80), (230, 0, 40), (145, 155, 165)),      # LATAM - Indigo, Coral, Gray
    }
    
    # Check full prefix match
    if prefix in airline_colors:
        return airline_colors[prefix]
    
    # Check 2-letter prefix for some airlines
    prefix2 = callsign[:2].upper()
    if prefix2 in airline_colors:
        return airline_colors[prefix2]
    
    # Default colors based on hash for variety
    hash_val = hash(prefix) % 360
    import colorsys
    r, g, b = colorsys.hsv_to_rgb(hash_val / 360, 0.8, 0.9)
    primary = (int(r * 255), int(g * 255), int(b * 255))
    
    # Generate complementary colors
    r2, g2, b2 = colorsys.hsv_to_rgb((hash_val + 120) % 360 / 360, 0.7, 0.85)
    secondary1 = (int(r2 * 255), int(g2 * 255), int(b2 * 255))
    
    r3, g3, b3 = colorsys.hsv_to_rgb((hash_val + 240) % 360 / 360, 0.6, 0.8)
    secondary2 = (int(r3 * 255), int(g3 * 255), int(b3 * 255))
    
    return (primary, secondary1, secondary2)


def draw_plane_shadow(draw: ImageDraw.ImageDraw, x: int, y: int,
                      heading: float | None):
    """Draw a subtle shadow beneath the plane for depth. Returns center position."""
    angle = math.radians((heading or 0) - 90)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    
    # Shadow is at ground position
    sx, sy = x, y
    
    # Plane scale factor
    scale = 2.5
    
    def rotate(dx, dy):
        return (sx + int(dx * scale * cos_a - dy * scale * sin_a),
                sy + int(dx * scale * sin_a + dy * scale * cos_a))
    
    nose = rotate(10, 0)
    tail = rotate(-8, 0)
    wing_left = rotate(0, -8)
    wing_right = rotate(0, 8)
    wing_front = rotate(2, 0)
    tail_top = rotate(-8, -3)
    tail_bottom = rotate(-8, 3)
    
    # Draw shadow - dark and subtle
    draw.line([wing_left, wing_front, wing_right], fill=(30, 30, 30), width=8)
    draw.line([nose, tail], fill=(30, 30, 30), width=10)
    draw.line([tail_top, tail, tail_bottom], fill=(30, 30, 30), width=7)
    
    # Return center position of shadow for connecting line
    return (sx, sy)


def draw_connecting_line(draw: ImageDraw.ImageDraw, shadow_pos: tuple[int, int], 
                        plane_pos: tuple[int, int], color: tuple[int, int, int]):
    """Draw a thin vertical line connecting shadow to plane."""
    # Draw a thin line from shadow to plane
    draw.line([shadow_pos, plane_pos], fill=color, width=1)


def draw_plane(draw: ImageDraw.ImageDraw, x: int, y: int,
               heading: float | None, callsign: str, altitude: float | None = None):
    """Draw a larger plane with airline-specific 3-color scheme at altitude-adjusted position."""
    angle = math.radians((heading or 0) - 90)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    
    # Calculate altitude offset (planes higher in the sky appear further up on screen)
    altitude_offset = 0
    if altitude is not None and altitude > 0:
        # Scale altitude to pixels - significantly elevated
        altitude_offset = int(altitude * ALTITUDE_SCALE)
    
    # Adjust y position based on altitude (negative = upward)
    y = y - altitude_offset
    
    # Plane scale factor (2.5x bigger)
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
    
    # Add secondary color accents on wingtips (small highlights at the tips only)
    wingtip_size = int(1.2 * scale)
    draw.ellipse([wing_left[0]-wingtip_size, wing_left[1]-wingtip_size, 
                  wing_left[0]+wingtip_size, wing_left[1]+wingtip_size],
                 fill=secondary1_color)
    draw.ellipse([wing_right[0]-wingtip_size, wing_right[1]-wingtip_size, 
                  wing_right[0]+wingtip_size, wing_right[1]+wingtip_size],
                 fill=secondary1_color)
    
    # Add secondary2 color accent on tail (small line matching tail shape)
    draw.line([tail_top, tail_bottom], fill=secondary2_color, width=int(1.5 * scale))
    
    # Draw callsign label
    label = (callsign or "").strip()
    if label:
        draw.text((x + int(12 * scale), y - int(10 * scale)), label,
                  fill=primary_color, stroke_fill=(0, 0, 0), stroke_width=2)
    
    # Return the plane's primary color for the connecting line
    return primary_color


def warp_to_trapezium(image, top_shrink, vertical_shift):
    """
    Warps an image so it looks like it is viewed from the side.

    Parameters
    ----------
    image : numpy array (OpenCV image)
    top_shrink : float
        How narrow the top becomes (0.5–0.8 works well)
    vertical_shift : float
        How much the top edge moves downward

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

    # Warp with transparent background
    warped = cv2.warpPerspective(image, matrix, (w, h), 
                                  borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=(0, 0, 0, 0))

    return warped


def create_hologram_cross(img: Image.Image) -> Image.Image:
    """
    Create a hologram cross layout with 4 copies of the image,
    each rotated to face toward the center.
    
    Layout:
           [Top - 180°]
    [Left - 90°] [Right - 270°]
         [Bottom - 0°]
    """
    # Ensure input is RGBA
    if img.mode != 'RGBA':
        img = img.convert('RGBA')
    
    width, height = img.size
    
    # Pre-rotate left and right to know their dimensions, then mirror vertically (since they're rotated)
    left = img.rotate(90, expand=True, fillcolor=(0, 0, 0, 0)).transpose(Image.FLIP_TOP_BOTTOM)
    right = img.rotate(-90, expand=True, fillcolor=(0, 0, 0, 0)).transpose(Image.FLIP_TOP_BOTTOM)
    
    # Minimal gap between images
    gap_x = 20
    gap_y = 800
    
    # Calculate canvas dimensions to tightly fit all rotated images
    # Horizontal: need to fit left + top/bottom + right side-by-side
    # Note: left and right are rotated, so their widths are the original height
    canvas_width = left.width + gap_x + width + gap_x + right.width
    
    # Vertical: need to fit top + bottom stacked, or accommodate left/right height
    vertical_for_topbottom = height + gap_y + height
    vertical_for_sides = max(left.height, right.height)
    canvas_height = max(vertical_for_topbottom, vertical_for_sides)
    
    canvas = Image.new('RGBA', (canvas_width, canvas_height), (0, 0, 0, 255))
    
    # Calculate center position for top/bottom images
    center_x = left.width + gap_x
    
    # All images are flipped 180° for hologram pyramid viewing, then mirrored
    # Bottom: 180° (upside down) - at bottom of vertical space
    bottom = img.rotate(180, expand=False, fillcolor=(0, 0, 0, 0)).transpose(Image.FLIP_LEFT_RIGHT)
    bottom_y = canvas_height - height
    canvas.paste(bottom, (center_x, bottom_y), bottom)
    
    # Top: 0° - at top of vertical space, mirrored
    top = img.copy().transpose(Image.FLIP_LEFT_RIGHT)
    canvas.paste(top, (center_x, 0), top)
    
    # Left: 90° - pushed to left edge, centered vertically
    left_y = (canvas_height - left.height) // 2
    canvas.paste(left, (0, left_y), left)
    
    # Right: -90° (270°) - pushed to right edge, centered vertically
    right_y = (canvas_height - right.height) // 2
    right_x = canvas_width - right.width
    canvas.paste(right, (right_x, right_y), right)
    
    # Convert back to RGB for display (composite onto black)
    final = Image.new('RGB', (canvas_width, canvas_height), (0, 0, 0))
    final.paste(canvas, (0, 0), canvas)
    
    return final


def composite(base_map: Image.Image, planes: list,
              lat: float, lon: float, zoom: int) -> Image.Image:
    # Use fixed vertical expansion based on MAX_ALTITUDE to keep canvas size constant
    vertical_expansion = int(MAX_ALTITUDE * ALTITUDE_SCALE) + 100  # Extra padding
    expanded_height = IMAGE_HEIGHT + vertical_expansion
    
    # Create expanded canvas with map at the bottom
    expanded_img = Image.new('RGB', (IMAGE_WIDTH, expanded_height), (0, 0, 0))
    expanded_img.paste(base_map, (0, vertical_expansion))
    
    # Apply slight blur to map for depth of field effect
    expanded_img = expanded_img.filter(ImageFilter.GaussianBlur(radius=0.5))
    draw = ImageDraw.Draw(expanded_img)

    # Store shadow positions and plane data for three-pass rendering
    plane_data = []
    
    # First pass: draw all shadows and collect positions (adjusted for expanded canvas)
    for state in planes:
        try:
            plon, plat = state[5], state[6]
            altitude = state[7]  # barometric altitude in meters
            if plon is None or plat is None:
                continue
            # Cap altitude at MAX_ALTITUDE
            if altitude is not None and altitude > MAX_ALTITUDE:
                altitude = MAX_ALTITUDE
            x, y = geo_to_pixel(plon, plat, lat, lon, zoom)
            # Adjust y for expanded canvas
            y_adjusted = y + vertical_expansion
            if 0 <= x < IMAGE_WIDTH and 0 <= y < IMAGE_HEIGHT:
                shadow_pos = draw_plane_shadow(draw, x, y_adjusted, state[10])
                plane_data.append({
                    'shadow_pos': shadow_pos,
                    'ground_pos': (x, y_adjusted),
                    'heading': state[10],
                    'callsign': state[1] or "",
                    'altitude': altitude
                })
        except (IndexError, TypeError):
            continue
    
    # Second pass: draw connecting lines
    for data in plane_data:
        altitude = data['altitude']
        if altitude is not None and altitude > 0:
            # Calculate plane position with altitude offset
            altitude_offset = int(altitude * ALTITUDE_SCALE)
            plane_pos = (data['ground_pos'][0], data['ground_pos'][1] - altitude_offset)
            
            # Get the plane's primary color for the line
            primary_color, _, _ = get_airline_color(data['callsign'])
            
            # Draw connecting line
            draw_connecting_line(draw, data['shadow_pos'], plane_pos, primary_color)
    
    # Third pass: draw all planes on top
    for data in plane_data:
        x, y = data['ground_pos']
        plane_color = draw_plane(draw, x, y, data['heading'], data['callsign'], data['altitude'])

    # Convert PIL Image to numpy array for OpenCV
    img_array = np.array(expanded_img)
    
    # Apply trapezoid perspective transformation with strong perspective
    warped_array = warp_to_trapezium(img_array, top_shrink=0.3, vertical_shift=0.8)
    
    # Convert back to PIL Image (RGBA with transparency)
    warped_img = Image.fromarray(warped_array, 'RGBA')
    
    # Find the bounding box of actual content (non-black pixels)
    # Convert to numpy for analysis
    warped_array_rgba = np.array(warped_img)
    
    # Find rows that have any non-black content
    # Sum across width and channels, any row with sum > threshold has content
    row_sums = np.sum(warped_array_rgba, axis=(1, 2))
    non_empty_rows = np.where(row_sums > 100)[0]  # Threshold to ignore near-black
    
    if len(non_empty_rows) > 0:
        # Crop to just the content area with minimal padding
        top_crop = max(0, non_empty_rows[0])  # No padding above topmost content
        bottom_crop = min(warped_img.height, non_empty_rows[-1])  # Minimal padding below
        
        # Further limit the height - crop more of the top if still too tall
        max_display_height = 700  # Maximum height in pixels
        cropped_height = bottom_crop - top_crop
        if cropped_height > max_display_height:
            # Keep bottom, crop more from top
            top_crop = bottom_crop - max_display_height
        
        # Crop the image
        warped_img = warped_img.crop((0, top_crop, warped_img.width, bottom_crop))
    
    # Create hologram cross layout
    return create_hologram_cross(warped_img)


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Plane tracker")
    parser.add_argument("--lat",  type=float, default=41.978611, help="Center latitude")
    parser.add_argument("--lon",  type=float, default=-87.904724, help="Center longitude")
    parser.add_argument("--zoom", type=int,   default=11,        help="Zoom level (1-12)")
    args = parser.parse_args()

    token = os.getenv("MAPBOX_TOKEN")
    
    # Initialize OpenSky token manager
    client_id = os.getenv("clientId")
    client_secret = os.getenv("clientSecret")
    token_manager = TokenManager(client_id, client_secret)
    
    # Fetch map once (doesn't change)
    print("Fetching map…")
    base_map = fetch_map(args.lat, args.lon, args.zoom, token)
    
    # Setup matplotlib for live updates with maximized window
    plt.ion()  # Turn on interactive mode
    # Create figure with moderate size
    fig = plt.figure(figsize=(14, 10))  # Reduced from 20x14 for smaller cross
    ax = fig.add_subplot(111)
    ax.axis('off')
    fig.patch.set_facecolor('black')
    fig.tight_layout(pad=0)  # Remove padding
    
    # Try to maximize window
    mng = plt.get_current_fig_manager()
    try:
        mng.frame.Maximize(True)  # wxPython
    except:
        try:
            mng.window.showMaximized()  # Qt
        except:
            pass  # Fallback: just use the large figsize
    
    plt.show()
    
    img_display = None
    
    try:
        while True:
            # Fetch plane data
            print("Fetching planes…")
            planes = fetch_planes(*bounding_box(args.lat, args.lon, args.zoom), token_manager)
            print(f"{len(planes)} aircraft found: {', '.join([plane[1] or 'N/A' for plane in planes[:5]])}")
            
            # Generate composite image
            result = composite(base_map, planes, args.lat, args.lon, args.zoom)
            
            # Update display
            if img_display is None:
                img_display = ax.imshow(np.array(result))
            else:
                img_display.set_data(np.array(result))
            
            plt.draw()
            plt.pause(5.0)  # Wait 5 seconds before next update
            
    except KeyboardInterrupt:
        print("\nShutting down...")
        plt.close()


if __name__ == "__main__":
    main()
