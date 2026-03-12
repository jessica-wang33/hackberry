"""
Fetch aircraft data within a geographical range using the OpenSky Network API
"""

import requests
from typing import List, Dict, Optional
import json


def get_planes_in_range(
    latitude: float,
    longitude: float,
    x_range: float,
    y_range: float
) -> List[Dict]:
    """
    Args:
        latitude: Center latitude in decimal degrees
        longitude: Center longitude in decimal degrees
        x_range: Longitude range in degrees (half-width of bounding box)
        y_range: Latitude range in degrees (half-height of bounding box)
    
    Returns:
        List of dictionaries containing aircraft data
    """
    
    # Calculate bounding box
    lamin = latitude - y_range
    lamax = latitude + y_range
    lomin = longitude - x_range
    lomax = longitude + x_range
    
    # OpenSky Network API endpoint
    url = "https://opensky-network.org/api/states/all"
    
    # Parameters for bounding box query
    params = {
        'lamin': lamin,
        'lomin': lomin,
        'lamax': lamax,
        'lomax': lomax
    }
    
    try:
        # Make API request
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        # Parse the response
        if 'states' not in data or data['states'] is None:
            print("No aircraft found in the specified range.")
            return []
        
        # Convert state vectors to dictionaries
        aircraft_list = []
        for state in data['states']:
            aircraft = {
                'icao24': state[0],           # Unique ICAO 24-bit address
                'callsign': state[1].strip() if state[1] else None,  # Callsign
                'origin_country': state[2],   # Country of origin
                'time_position': state[3],    # Unix timestamp of last position update
                'last_contact': state[4],     # Unix timestamp of last contact
                'longitude': state[5],        # Longitude in decimal degrees
                'latitude': state[6],         # Latitude in decimal degrees
                'baro_altitude': state[7],    # Barometric altitude in meters
                'on_ground': state[8],        # Boolean - on ground or not
                'velocity': state[9],         # Velocity in m/s
                'true_track': state[10],      # True track in degrees (north=0°)
                'vertical_rate': state[11],   # Vertical rate in m/s
                'sensors': state[12],         # IDs of sensors that received messages
                'geo_altitude': state[13],    # Geometric altitude in meters
                'squawk': state[14],          # Transponder code
                'spi': state[15],             # Special purpose indicator
                'position_source': state[16]  # Origin of position (0=ADS-B, 1=ASTERIX, 2=MLAT)
            }
            aircraft_list.append(aircraft)
        
        return aircraft_list
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from OpenSky API: {e}")
        return []


def print_aircraft_info(aircraft_list: List[Dict]) -> None:
    """
    Args:
        aircraft_list: List of aircraft dictionaries
    """
    if not aircraft_list:
        print("No aircraft to display.")
        return
    
    print(f"\nFound {len(aircraft_list)} aircraft:\n")
    print("-" * 100)
    
    for aircraft in aircraft_list:
        callsign = aircraft['callsign'] or 'N/A'
        lat = aircraft['latitude'] or 'N/A'
        lon = aircraft['longitude'] or 'N/A'
        altitude = aircraft['baro_altitude']
        altitude_str = f"{altitude:.0f}m" if altitude else 'N/A'
        velocity = aircraft['velocity']
        velocity_str = f"{velocity:.1f} m/s" if velocity else 'N/A'
        country = aircraft['origin_country']
        on_ground = "Ground" if aircraft['on_ground'] else "Airborne"
        
        print(f"Callsign: {callsign:10} | Country: {country:15} | Status: {on_ground:8}")
        print(f"Position: {lat:10}, {lon:10} | Altitude: {altitude_str:10} | Speed: {velocity_str}")
        print("-" * 100)


if __name__ == "__main__":
    center_lat = 40.4439
    center_lon = -79.9428
    x_range = 0.1
    y_range = 0.1
    
    print(f"Fetching aircraft data near LAX...")
    print(f"Center: ({center_lat}, {center_lon})")
    print(f"Range: ±{x_range}° longitude, ±{y_range}° latitude")
    
    planes = get_planes_in_range(center_lat, center_lon, x_range, y_range)
    print_aircraft_info(planes)
