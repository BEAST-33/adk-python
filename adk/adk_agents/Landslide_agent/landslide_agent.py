import math
import requests
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from geopy.geocoders import Nominatim
from datetime import datetime, timedelta

from fastapi import FastAPI, Request, HTTPException
import uvicorn

import logging
OPENWEATHER_API_KEY = "917833134fe9d775fcc50d2dbfc9abe3"
ELEVATION_API = "https://api.opentopodata.org/v1/aster30m?locations={lat},{lon}"

CASSANDRA_HOSTS = ["10.5.10.190"] 
CASSANDRA_PORT = 9042
CASSANDRA_USER = "cassandra"
CASSANDRA_PASSWORD = "cassandra"
CASSANDRA_KEYSPACE = "ai_bus_info"
CASSANDRA_TABLE = "landslide_data"

geolocator = Nominatim(user_agent="geoapi")

app = FastAPI()

cluster = None
session = None
def get_coordinates(location_name):
    try:
        location = geolocator.geocode(location_name)
        if location:
            return location.latitude, location.longitude
    except:
        pass
    return None, None

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def interpolate_points(lat1, lon1, lat2, lon2, num_points=50):
    return [(lat1 + i/num_points * (lat2 - lat1), lon1 + i/num_points * (lon2 - lon1)) for i in range(num_points + 1)]

def fetch_city_data_from_cassandra():
    global session
    rows = session.execute(f"SELECT district_name, state_name, latitude, longitude, rank FROM {CASSANDRA_KEYSPACE}.{CASSANDRA_TABLE}")
    city_data = [{
        'District': r.district_name,
        'State': r.state_name,
        'Latitude': r.latitude,
        'Longitude': r.longitude,
        'Rank': getattr(r, 'rank', None)
    } for r in rows if r.latitude and r.longitude]
    return city_data

def find_nearest_cities_to_route(from_city, to_city, city_data, max_cities=5, max_distance_km=200):
    from_lat, from_lon = get_coordinates(from_city)
    to_lat, to_lon = get_coordinates(to_city)
    if None in (from_lat, from_lon, to_lat, to_lon):
        return []

    route_points = interpolate_points(from_lat, from_lon, to_lat, to_lon)
    city_distances = []

    for city in city_data:
        try:
            lat, lon = float(city['Latitude']), float(city['Longitude'])
            min_dist = float('inf')
            closest_point = None
            closest_index = -1
            for i, (r_lat, r_lon) in enumerate(route_points):
                dist = haversine(lat, lon, r_lat, r_lon)
                if dist < min_dist:
                    min_dist = dist
                    closest_point = (r_lat, r_lon)
                    closest_index = i

            if min_dist <= max_distance_km:
                try:
                    loc = geolocator.reverse(closest_point, timeout=10)
                    location_name = loc.address if loc else f"Route Point #{closest_index}"
                except:
                    location_name = f"Route Point #{closest_index}"

                city['Distance'] = f"{round(min_dist, 2)} km from {location_name}"
                city['ClosestRoutePoint'] = {'lat': closest_point[0], 'lon': closest_point[1], 'index': closest_index}
                city_distances.append(city)
        except:
            continue

    city_distances.sort(key=lambda x: float(x['Distance'].split()[0]))  # sort by numeric distance
    return city_distances[:max_cities]


def get_forecast(lat, lon):
    forecast_url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric"
    try:
        resp = requests.get(forecast_url)
        if resp.status_code == 200:
            return resp.json()
    except:
        pass
    return None

def get_yesterday_rain(lat, lon):
    dt = int((datetime.utcnow() - timedelta(days=1)).timestamp())
    url = f"https://api.openweathermap.org/data/2.5/onecall/timemachine?lat={lat}&lon={lon}&dt={dt}&appid={OPENWEATHER_API_KEY}&units=metric"
    try:
        resp = requests.get(url)
        if resp.status_code == 200:
            data = resp.json()
            return sum(hour.get('rain', {}).get('1h', 0) for hour in data.get('hourly', []))
    except:
        pass
    return 0

def get_forecast_rain_by_hours(forecast_data, hours):
    if not forecast_data or "list" not in forecast_data:
        return 0.0
    num_entries = int(hours / 3)
    return sum(entry.get('rain', {}).get('3h', 0) for entry in forecast_data['list'][:num_entries])

def get_elevation(lat, lon):
    try:
        response = requests.get(ELEVATION_API.format(lat=lat, lon=lon))
        if response.status_code == 200:
            result = response.json().get("results")
            if result and result[0]["elevation"] is not None:
                return result[0]["elevation"]
    except:
        pass
    return None

def categorize_rainfall(value):
    if value < 1:
        return f"🌤️ No Rain (0 mm)"
    elif value < 5:
        return f"🌦️ Light Rain (0-5 mm)"
    elif value < 15:
        return f"🌧️ Moderate Rain (5-15 mm)"
    elif value < 30:
        return f"🌩️ Heavy Rain (15-30 mm)"
    else:
        return f"⛈️ Very Heavy Rain (30+ mm)"

def assess_landslide_risk(city, lat, lon):
    forecast = get_forecast(lat, lon)
    past_rain = get_yesterday_rain(lat, lon)
    elevation = get_elevation(lat, lon)

    rain_4h = get_forecast_rain_by_hours(forecast, 4)
    rain_10h = get_forecast_rain_by_hours(forecast, 10)
    rain_24h = get_forecast_rain_by_hours(forecast, 24)

    total_rainfall = past_rain + rain_24h

    # Determine risk level
    if elevation is None:
        risk = "❓ Unknown"
        score = 0
    elif elevation >= 800 and total_rainfall > 30:
        risk = "🔴 High"
        score = 3
    elif elevation >= 500 and total_rainfall > 15:
        risk = "🟠 Moderate"
        score = 2
    else:
        risk = "🟢 Low"
        score = 1

    return {
        "city": city,
        "rain_past": categorize_rainfall(past_rain),
        "rain_4h": categorize_rainfall(rain_4h),
        "rain_10h": categorize_rainfall(rain_10h),
        "rain_24h": categorize_rainfall(rain_24h),
        "rain_total": categorize_rainfall(total_rainfall),
        "elevation": f"{round(elevation, 2)} m" if elevation else "N/A",
        "risk": risk,
        "score": score
    }


class LandslideRiskAgent:
    def __init__(self):
        pass

    def analyze_route(self, from_city, to_city):
        city_data = fetch_city_data_from_cassandra()
        if not city_data:
            return []

        route_cities = find_nearest_cities_to_route(from_city, to_city, city_data)

        results = []
        for city in route_cities:
            lat, lon = float(city['Latitude']), float(city['Longitude'])
            risk_data = assess_landslide_risk(f"{city['District']} ({city['State']})", lat, lon)
            result = {**city, **risk_data}
            results.append(result)

        results.sort(key=lambda x: (-x['score'], x.get('Rank') or float('inf')))
        return results


landslide_agent = None

@app.on_event("startup")
async def startup_event():
    global cluster, session, landslide_agent
    auth_provider = PlainTextAuthProvider(username=CASSANDRA_USER, password=CASSANDRA_PASSWORD)
    cluster = Cluster(contact_points=CASSANDRA_HOSTS, port=CASSANDRA_PORT, auth_provider=auth_provider)
    session = cluster.connect()
    landslide_agent = LandslideRiskAgent()

@app.on_event("shutdown")
async def shutdown_event():
    if session:
        session.shutdown()
    if cluster:
        cluster.shutdown()

def make_a2a_response(request_id, task_id, state, message_text, history=None, artifacts=None):
    now = datetime.utcnow().isoformat()
    result = {
        "id": task_id,
        "status": {
            "state": state,
            "timestamp": now
        },
        "history": history or []
    }
    if state == "completed":
        result["artifacts"] = artifacts or [{
            "parts": [{"type": "text", "text": message_text}],
            "index": 0
        }]
    else:
        result["status"]["message"] = {
            "role": "agent",
            "parts": [{"type": "text", "text": message_text}]
        }
    return {"jsonrpc": "2.0", "id": request_id, "result": result}

@app.post("/")
async def root(request: Request):
    payload = await request.json()
    logging.debug(f"Received payload: {payload}")
    
    params = payload.get("params", payload)
    task_id = params.get("id")
    request_id = payload.get("id") or params.get("id")
    message = params.get("message", {})
    parts = message.get("parts", [])

    if isinstance(parts, list) and parts:
        user_text = parts[0].get("text", "") if isinstance(parts[0], dict) else str(parts[0])
    elif isinstance(parts, dict):
        user_text = parts.get("text", "")
    else:
        user_text = ""

    if not user_text:
        error_message = "Input is missing."
        logging.error(error_message)
        raise HTTPException(status_code=400, detail=error_message)

    user_input_lower = user_text.lower()
    if "analyse landslide risk for" not in user_input_lower:
        example_text = (
            "❌ Invalid format.\n\n"
            "✅ Examples of valid inputs:\n"
            "- Analyse landslide risk for Bangalore to Delhi\n"
            "- Analyse the landslide risk for the route Bangalore to Delhi\n"
            "- Analyse landslide risk for Shimla to Manali\n"
            "- Analyse landslide risk for Manali to Leh\n\n"
            "Please enter the route in similar format."
        )
        return make_a2a_response(request_id, task_id, "in_progress", example_text)

    try:
        after_phrase = user_input_lower.split("analyse landslide risk for", 1)[1].strip()
        parts_route = after_phrase.split("to")
        if len(parts_route) != 2:
            raise ValueError("Invalid format")
        from_city = parts_route[0].strip().title()
        to_city = parts_route[1].strip().title()
    except Exception as e:
        logging.error(f"Parsing error: {e}")
        return make_a2a_response(
            request_id, task_id, "failed",
            "Invalid format. Please use: 'Analyse landslide risk for <city1> to <city2>'."
        )
    
    results = landslide_agent.analyze_route(from_city, to_city)
    if not results:
        return make_a2a_response(request_id, task_id, "completed", "No high-risk cities found on route.")

    summary = "\n".join(
    f"{r['city']} | Elev: {r['elevation']}m | Dist: {r.get('Distance', 'N/A')} | Rain (4h): {r['rain_4h']} mm | Rain (10h): {r['rain_10h']} mm | Rain (24h): {r['rain_24h']} mm | Risk: {r['risk']} "
    for r in results
)


    most_risky = results[0]
    summary += f"\n\n🚨 Most Risk-Prone: {most_risky['city']} ({most_risky['risk']})"

    return make_a2a_response(request_id, task_id, "completed", summary)

@app.post("/tasks/send")
async def tasks_send(request: Request):
    return await root(request)

@app.get("/.well-known/agent.json")
async def agent_manifest(request: Request):
    logging.info(f"Request received: {request.method} {request.url}")
    logging.info(f"Request headers: {dict(request.headers)}")

    response_content = {
        "name": "Landslide Risk Agent",
        "description": (
            "Analyzes landslide risk for cities along a given route using weather forecasts, "
            "elevation data, and historical rainfall stored in Cassandra. "
            "Returns risk summaries and identifies the most risk-prone location."
        ),
        "url": "http://10.5.10.190:8023/",
        "version": "1.0.1",
        "provider": {
            "organization": "redBus",
            "url": "https://redbus.in"
        },
        "capabilities": {
            "streaming": False,
            "pushNotifications": False,
            "stateTransitionHistory": False
        },
        "defaultInputModes": ["text", "text/plain"],
        "defaultOutputModes": ["text", "text/plain"],
        "skills": [
            {
                "id": "analyze_landslide_risk",
                "name": "Analyze Landslide Risk",
                "description": (
                    "Returns landslide risk information for cities along a route. "
                    "Uses input in the format 'Analyse landslide risk for <city1> to <city2>'."
                ),
                "tags": ["landslide", "risk", "weather", "elevation", "routing"],
                "examples": [
                    "Analyse landslide risk for Bangalore to Delhi",
                    "Analyse the landslide risk for the route Bangalore to Delhi",
                    "Analyse landslide risk for Shimla to Manali",
                    "Analyse landslide risk for Manali to Leh"
                ],
                "inputModes": ["text", "text/plain"],
                "outputModes": ["text", "text/plain"],
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "object",
                            "properties": {
                                "parts": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "type": {"type": "string", "enum": ["text"]},
                                            "text": {
                                                "type": "string",
                                                "description": "Route input in the format 'Analyse landslide risk for <city1> to <city2>'."
                                            }
                                        },
                                        "required": ["type", "text"]
                                    }
                                }
                            },
                            "required": ["parts"]
                        }
                    },
                    "required": ["message"]
                },
                "output_schema": {
                    "type": "object",
                    "properties": {
                        "summary": {
                            "type": "string",
                            "description": "Formatted risk report for cities along the route."
                        },
                        "most_risk_prone": {
                            "type": "string",
                            "description": "Name of the city with the highest landslide risk."
                        }
                    },
                    "required": ["summary", "most_risk_prone"]
                }
            }
        ]
    }

    return response_content

if __name__ == "__main__":
    uvicorn.run("landslide-a2a-agent:app", host="10.5.10.190", port=8023,reload=True)
