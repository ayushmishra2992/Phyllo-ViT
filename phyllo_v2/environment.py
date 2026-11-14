"""Ambient air-quality retrieval from OpenAQ (v3). Requires a user-supplied API key (env OPENAQ_API_KEY).

OpenAQ aggregates government monitors; licences are per source (read from each location's response, never assumed).
Everything returned here is an AMBIENT reading from a monitoring station - NOT something measured from the leaf photo, and
it may not represent the exact leaf location (station distance is reported).
"""
import math
import os
from datetime import datetime, timezone

import httpx

from phyllo.aqi_standard import compute_aqi

from . import AVAILABLE, UNAVAILABLE, USER_INPUT_REQUIRED, section

BASE = os.environ.get("OPENAQ_BASE_URL", "https://api.openaq.org/v3")
SEARCH_RADIUS_M = int(os.environ.get("OPENAQ_RADIUS_M", "25000"))   # OpenAQ v3 maximum radius is 25 km
NAMES = {"pm25": "PM2.5", "pm10": "PM10", "no2": "NO2", "co": "CO", "o3": "O3", "so2": "SO2"}
WANTED = set(NAMES)


def haversine_km(lat1, lon1, lat2, lon2):
    p = math.pi / 180
    a = (math.sin((lat2 - lat1) * p / 2) ** 2 + math.cos(lat1 * p) * math.cos(lat2 * p) * math.sin((lon2 - lon1) * p / 2) ** 2)
    return 6371.0088 * 2 * math.asin(math.sqrt(a))


def _get(client, path, key, **params):
    r = client.get(f"{BASE}{path}", params=params, headers={"X-API-Key": key}, timeout=20)
    if r.status_code == 429:
        raise RuntimeError("OpenAQ rate limit reached (HTTP 429); try again later.")
    if r.status_code in (401, 403):
        raise RuntimeError(f"OpenAQ rejected the API key (HTTP {r.status_code}).")
    r.raise_for_status()
    return r.json()


def fetch_ambient(lat, lon, collected_at=None, client=None, api_key=None, max_stations=3):
    """Return an environment section. `client` may be injected (tests use a mock transport and say so)."""
    if lat is None or lon is None:
        return section(USER_INPUT_REQUIRED, "Provide a location (latitude/longitude) to look up nearby monitoring stations.")
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return section(USER_INPUT_REQUIRED, "Latitude must be in [-90, 90] and longitude in [-180, 180].")
    key = api_key or os.environ.get("OPENAQ_API_KEY", "")
    if not key:
        return section(USER_INPUT_REQUIRED, "No OpenAQ API key configured (set OPENAQ_API_KEY; free registration at explore.openaq.org/register). "
                       "No ambient data is shown.")
    own = client is None
    client = client or httpx.Client()
    try:
        locs = _get(client, "/locations", key, coordinates=f"{lat},{lon}", radius=SEARCH_RADIUS_M, limit=20)
        results = locs.get("results", [])
        if not results:
            return section(UNAVAILABLE, f"No OpenAQ monitoring station within {SEARCH_RADIUS_M // 1000} km of this location.")
        for r in results:
            c = r.get("coordinates") or {}
            r["_dist"] = haversine_km(lat, lon, c.get("latitude", lat), c.get("longitude", lon)) if c else 1e9
        results.sort(key=lambda r: r["_dist"])
        stations = []
        for loc in results:
            if len(stations) >= max_stations:
                break
            sensors = {s["id"]: s for s in loc.get("sensors", []) if (s.get("parameter") or {}).get("name") in WANTED}
            if not sensors:
                continue
            latest = _get(client, f"/locations/{loc['id']}/latest", key).get("results", [])
            readings = []
            for item in latest:
                s = sensors.get(item.get("sensorsId"))
                if not s:
                    continue
                p = s["parameter"]
                dt = (item.get("datetime") or {}).get("utc")
                readings.append({"parameter": NAMES[p["name"]], "value": item.get("value"), "unit": p.get("units"), "timestamp_utc": dt})
            if not readings:
                continue
            stations.append({
                "station_id": loc["id"], "name": loc.get("name"), "distance_km": round(loc["_dist"], 2),
                "provider": (loc.get("provider") or {}).get("name"), "licences": [l.get("name") for l in (loc.get("licenses") or [])],
                "readings": readings})
        if not stations:
            return section(UNAVAILABLE, "Stations were found nearby but none reported PM2.5/PM10/NO2/CO/O3/SO2 latest values.")
        best = stations[0]
        indicative = None
        conc = {}
        for r in best["readings"]:
            if r["value"] is None or r["value"] < 0:
                continue
            if r["parameter"] in ("PM2.5", "PM10", "NO2", "O3") and r["unit"] in ("µg/m³", "ug/m3"):
                conc[r["parameter"]] = r["value"]
            elif r["parameter"] == "CO" and r["unit"] in ("µg/m³", "ug/m3"):
                conc["CO"] = r["value"] / 1000.0   # CPCB CO unit is mg/m3
        if conc:
            try:
                a = compute_aqi(conc)
                indicative = {"aqi": a["aqi"], "category": a["category"], "dominant_pollutant": a.get("dominant_pollutant"),
                              "note": "CPCB-formula index applied to the LATEST single readings. The official AQI uses 24-hour averages, "
                                      "so this is indicative only and not an official AQI."}
            except ValueError:
                indicative = None
        note = None
        if collected_at:
            note = (f"Station values are the latest available readings, not the readings at the supplied collection time ({collected_at}). "
                    "Historical retrieval is not implemented.")
        return section(AVAILABLE, "Ambient readings from OpenAQ monitoring stations; not measured from the leaf and possibly not representative of the exact leaf location.",
                       source="OpenAQ API v3 (https://api.openaq.org/v3)", retrieved_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                       query={"lat": lat, "lon": lon, "radius_km": SEARCH_RADIUS_M // 1000}, nearest_station=best, other_stations=stations[1:],
                       indicative_index_from_latest=indicative, time_note=note,
                       representativeness_note=f"Nearest station is {best['distance_km']} km away; street-level and leaf-level exposure can differ substantially.")
    except Exception as e:  # network, auth, rate limit, malformed response
        return section(UNAVAILABLE, f"Could not retrieve ambient data: {e}")
    finally:
        if own:
            client.close()
