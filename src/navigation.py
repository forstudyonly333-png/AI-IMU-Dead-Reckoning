import requests

def geocode_place(place_name):
    """
    India ki kisi bhi location ya city ka exact Lat, Lon dhoondhta hai.
    """
    try:
        url = f"https://nominatim.openstreetmap.org/search?q={place_name}, India&format=json&limit=1"
        headers = {"User-Agent": "AI_IMU_DR_Navigator_SIH"}
        res = requests.get(url, headers=headers, timeout=3.0).json()
        if res and len(res) > 0:
            return float(res[0]["lat"]), float(res[0]["lon"]), res[0]["display_name"]
    except Exception:
        pass
    return None, None, None

def get_pan_india_route(start_lat, start_lon, dest_lat, dest_lon):
    """
    OSRM se real road network geometry aur driving distance fetch karta hai.
    """
    try:
        url = (
            f"http://router.project-osrm.org/route/v1/driving/"
            f"{start_lon},{start_lat};{dest_lon},{dest_lat}?"
            f"overview=full&geometries=geojson"
        )
        res = requests.get(url, timeout=4.0).json()
        if "routes" in res and len(res["routes"]) > 0:
            route_data = res["routes"][0]
            coords = [[c[1], c[0]] for c in route_data["geometry"]["coordinates"]]
            dist_km = route_data["distance"] / 1000.0
            duration_min = route_data["duration"] / 60.0
            return coords, dist_km, duration_min
    except Exception:
        pass
    # Fallback straight line
    return [[start_lat, start_lon], [dest_lat, dest_lon]], 0.0, 0.0