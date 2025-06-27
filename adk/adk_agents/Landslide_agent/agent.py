
import logging
import math
import requests
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from geopy.geocoders import Nominatim
from datetime import datetime, timedelta

from google.adk.agents import LlmAgent
from google.adk.models import Gemini # This import is not used in the provided code, but kept as is.
from google.adk.models.lite_llm import LiteLlm
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Constants (moved from landslide_agent.py)
OPENWEATHER_API_KEY = "917833134fe9d775fcc50d2dbfc9abe3"
ELEVATION_API = "https://api.opentopodata.org/v1/aster30m?locations={lat},{lon}"
CASSANDRA_HOSTS = ["10.5.10.190"]
CASSANDRA_PORT = 9042
CASSANDRA_USER = "cassandra"
CASSANDRA_PASSWORD = "cassandra"
CASSANDRA_KEYSPACE = "ai_bus_info"
CASSANDRA_TABLE = "landslide_data"

geolocator = Nominatim(user_agent="geoapi")

# Global Cassandra session (to be managed by agent initialization/shutdown if needed,
# or passed to the tool function)
cluster = None
session = None

def get_coordinates(location_name):
    """Get latitude and longitude for a given location name."""
    try:
        location = geolocator.geocode(location_name)
        if location:
            return location.latitude, location.longitude
    except Exception as e:
        logger.error(f"Error getting coordinates for {location_name}: {e}")
    return None, None

def haversine(lat1, lon1, lat2, lon2):
    """Calculate the distance between two points on Earth using the Haversine formula."""
    R = 6371  # Radius of Earth in kilometers
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def interpolate_points(lat1, lon1, lat2, lon2, num_points=50):
    """Interpolate points along a line between two geographical coordinates."""
    return [(lat1 + i/num_points * (lat2 - lat1), lon1 + i/num_points * (lon2 - lon1)) for i in range(num_points + 1)]

def fetch_city_data_from_cassandra():
    """Fetches city data from Cassandra."""
    global session
    if session is None:
        # Initialize Cassandra connection if not already established
        try:
            auth_provider = PlainTextAuthProvider(username=CASSANDRA_USER, password=CASSANDRA_PASSWORD)
            cluster = Cluster(contact_points=CASSANDRA_HOSTS, port=CASSANDRA_PORT, auth_provider=auth_provider)
            session = cluster.connect()
            session.set_keyspace(CASSANDRA_KEYSPACE)
            logger.info("Cassandra connection established.")
        except Exception as e:
            logger.error(f"Failed to connect to Cassandra: {e}")
            return []

    try:
        rows = session.execute(f"SELECT district_name, state_name, latitude, longitude, rank FROM {CASSANDRA_KEYSPACE}.{CASSANDRA_TABLE}")
        city_data = [{
            'District': r.district_name,
            'State': r.state_name,
            'Latitude': r.latitude,
            'Longitude': r.longitude,
            'Rank': getattr(r, 'rank', None)
        } for r in rows if r.latitude and r.longitude]
        return city_data
    except Exception as e:
        logger.error(f"Error fetching city data from Cassandra: {e}")
        return []

def find_nearest_cities_to_route(from_city, to_city, city_data, max_cities=5, max_distance_km=200):
    """Finds cities nearest to the route between two given cities."""
    from_lat, from_lon = get_coordinates(from_city)
    to_lat, to_lon = get_coordinates(to_city)
    if None in (from_lat, from_lon, to_lat, to_lon):
        logger.warning(f"Could not get coordinates for one or both cities: {from_city}, {to_city}")
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
                except Exception:
                    location_name = f"Route Point #{closest_index}"

                city['Distance'] = f"{round(min_dist, 2)} km from {location_name}"
                city['ClosestRoutePoint'] = {'lat': closest_point[0], 'lon': closest_point[1], 'index': closest_index}
                city_distances.append(city)
        except Exception as e:
            logger.error(f"Error processing city {city.get('District')}: {e}")
            continue

    city_distances.sort(key=lambda x: float(x['Distance'].split()[0]))
    return city_distances[:max_cities]

def get_forecast(lat, lon):
    """Fetches weather forecast from OpenWeatherMap."""
    forecast_url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric"
    try:
        resp = requests.get(forecast_url)
        resp.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
        return resp.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching forecast for {lat},{lon}: {e}")
    return None

def get_yesterday_rain(lat, lon):
    """Fetches yesterday's rainfall from OpenWeatherMap historical data."""
    dt = int((datetime.utcnow() - timedelta(days=1)).timestamp())
    url = f"https://api.openweathermap.org/data/2.5/onecall/timemachine?lat={lat}&lon={lon}&dt={dt}&appid={OPENWEATHER_API_KEY}&units=metric"
    try:
        resp = requests.get(url)
        resp.raise_for_status()
        data = resp.json()
        return sum(hour.get('rain', {}).get('1h', 0) for hour in data.get('hourly', []))
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching yesterday's rain for {lat},{lon}: {e}")
    return 0

def get_forecast_rain_by_hours(forecast_data, hours):
    """Calculates total forecast rain for a given number of hours."""
    if not forecast_data or "list" not in forecast_data:
        return 0.0
    num_entries = int(hours / 3) # OpenWeatherMap provides forecast in 3-hour steps
    return sum(entry.get('rain', {}).get('3h', 0) for entry in forecast_data['list'][:num_entries])

def get_elevation(lat, lon):
    """Fetches elevation data from OpenTopoData."""
    try:
        response = requests.get(ELEVATION_API.format(lat=lat, lon=lon))
        response.raise_for_status()
        result = response.json().get("results")
        if result and result[0]["elevation"] is not None:
            return result[0]["elevation"]
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching elevation for {lat},{lon}: {e}")
    return None

def categorize_rainfall(value):
    """Categorizes rainfall amount into descriptive strings."""
    if value < 1:
        return "🌤️ No Rain (0 mm)"
    elif value < 5:
        return "🌦️ Light Rain (0-5 mm)"
    elif value < 15:
        return "🌧️ Moderate Rain (5-15 mm)"
    elif value < 30:
        return "🌩️ Heavy Rain (15-30 mm)"
    else:
        return "⛈️ Very Heavy Rain (30+ mm)"

def assess_landslide_risk(city_name, lat, lon):
    """Assesses landslide risk for a given city and coordinates."""
    forecast = get_forecast(lat, lon)
    past_rain = get_yesterday_rain(lat, lon)
    elevation = get_elevation(lat, lon)

    rain_4h = get_forecast_rain_by_hours(forecast, 4)
    rain_10h = get_forecast_rain_by_hours(forecast, 10)
    rain_24h = get_forecast_rain_by_hours(forecast, 24)

    total_rainfall = past_rain + rain_24h

    risk = "❓ Unknown"
    score = 0
    if elevation is None:
        risk = "❓ Unknown (Elevation N/A)"
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
        "city": city_name,
        "rain_past": categorize_rainfall(past_rain),
        "rain_4h": categorize_rainfall(rain_4h),
        "rain_10h": categorize_rainfall(rain_10h),
        "rain_24h": categorize_rainfall(rain_24h),
        "rain_total": categorize_rainfall(total_rainfall),
        "elevation": f"{round(elevation, 2)} m" if elevation is not None else "N/A",
        "risk": risk,
        "score": score
    }

# Define the response model for the tool
class LandslideRiskAssessment(BaseModel):
    city: str
    rain_past: str = Field(description="Rainfall in the past 24 hours (categorized)")
    rain_4h: str = Field(description="Forecast rainfall for the next 4 hours (categorized)")
    rain_10h: str = Field(description="Forecast rainfall for the next 10 hours (categorized)")
    rain_24h: str = Field(description="Forecast rainfall for the next 24 hours (categorized)")
    rain_total: str = Field(description="Total recent and forecast rainfall (categorized)")
    elevation: str = Field(description="Elevation of the city in meters or 'N/A'")
    risk: str = Field(description="Categorized landslide risk (Low, Moderate, High, Unknown)")
    score: int = Field(description="Numeric score for landslide risk (0-3, higher is riskier)")
    distance_from_route: str = Field(None, description="Distance from the closest point on the route, if applicable")
    original_rank: int = Field(None, description="Original rank of the city from Cassandra data, if applicable")

class LandslideRiskResponse(BaseModel):
    status: str
    message: str
    results: list[LandslideRiskAssessment] = []


def landslide_risk_tool(from_city: str, to_city: str) -> LandslideRiskResponse:
    """
    Analyzes landslide risk for cities along a bus route between two specified cities.

    Args:
        from_city: The starting city of the route.
        to_city: The destination city of the route.

    Returns:
        LandslideRiskResponse: An object containing the status, a message, and a list of
                               landslide risk assessments for relevant cities.
    """
    logger.info(f"landslide_risk_tool called for route: {from_city} to {to_city}")

    if not from_city or not to_city:
        return LandslideRiskResponse(
            status="error",
            message="Both 'from_city' and 'to_city' must be provided.",
            results=[]
        )

    try:
        city_data = fetch_city_data_from_cassandra()
        if not city_data:
            return LandslideRiskResponse(
                status="error",
                message="Could not fetch city data from Cassandra. Please check connection.",
                results=[]
            )

        route_cities = find_nearest_cities_to_route(from_city, to_city, city_data)
        if not route_cities:
            return LandslideRiskResponse(
                status="not_found",
                message=f"No relevant cities found near the route from {from_city} to {to_city} for landslide risk assessment.",
                results=[]
            )

        assessments = []
        for city in route_cities:
            lat, lon = float(city['Latitude']), float(city['Longitude'])
            city_name_display = f"{city['District']} ({city['State']})"
            risk_data = assess_landslide_risk(city_name_display, lat, lon)
            assessment = LandslideRiskAssessment(
                city=risk_data['city'],
                rain_past=risk_data['rain_past'],
                rain_4h=risk_data['rain_4h'],
                rain_10h=risk_data['rain_10h'],
                rain_24h=risk_data['rain_24h'],
                rain_total=risk_data['rain_total'],
                elevation=risk_data['elevation'],
                risk=risk_data['risk'],
                score=risk_data['score'],
                distance_from_route=city.get('Distance'),
                original_rank=city.get('Rank')
            )
            assessments.append(assessment)

        # Sort results by score (descending) and then by original rank (ascending if available)
        assessments.sort(key=lambda x: (-x.score, x.original_rank if x.original_rank is not None else float('inf')))

        message_summary = "Landslide risk analysis completed."
        return LandslideRiskResponse(
            status="success",
            message=message_summary,
            results=assessments
        )

    except Exception as e:
        logger.error(f"Error in landslide_risk_tool for {from_city} to {to_city}: {e}", exc_info=True)
        return LandslideRiskResponse(
            status="error",
            message=f"An unexpected error occurred: {str(e)}. Please try again later.",
            results=[]
        )

# Create the LLM agent
Landslide_Risk_agent = LlmAgent(
    name="Landslide_Risk_Agent",
    model=LiteLlm(model="ollama_chat/llama3.2:latest"), # Ensure this model is accessible and configured correctly
    tools=[landslide_risk_tool], # Register the tool with the agent
    instruction="""
    You are a helpful assistant specialized in analyzing landslide risks along bus routes.
    Your primary goal is to assess and present landslide risk information for a given route, and then conclude your response for the current turn.

    **Strict Steps:**
    1.  **Extract Route Information:** Carefully identify and extract the 'from' city and 'to' city from the user's message. The expected format is "Analyse landslide risk for <city1> to <city2>".
    2.  **Call Tool:** Use the `landslide_risk_tool` function with the extracted 'from' and 'to' cities.
    3.  **Process Tool Result and Respond:**
        * If the tool returns a `status: "success"` and `results` are available, present a clear and concise summary of the landslide risks. Start by highlighting the most risk-prone location(s) and then list other relevant cities with their risk details (elevation, rainfall, risk level). After presenting the summary, **your task for this turn is complete, and you should provide a final, helpful response to the user.**
        * If the tool returns a `status: "not_found"`, inform the user that no relevant landslide risk data was found for that route, and politely offer to check another route. **Then, your task for this turn is complete.**
        * If the tool returns an `status: "error"`, acknowledge the error gracefully and suggest they try again later or with a different route. **Then, your task for this turn is complete.**
    4.  **Handle Missing Route Information/Invalid Format:** If the user does not provide the route in the expected format (e.g., "Analyse landslide risk for <city1> to <city2>"), politely ask them to specify the route in the correct format. Do NOT try to call the tool without proper 'from' and 'to' cities.

    **Important:** Once you have provided a response based on the `landslide_risk_tool`'s output (whether success, not found, or error), consider the current turn's objective achieved. **Do not re-prompt for the same route or re-call the tool for the same request.** Focus on providing a single, conclusive answer for each user query.
    """
)

# Set the Landslide_Risk_agent as the root agent for standalone execution
root_agent = Landslide_Risk_agent

# --- Functions for managing Cassandra connection for standalone execution (if needed) ---
# These functions would typically be called when the script starts/stops
# if you're running the ADK agent directly as a script that needs a persistent DB connection.
# In a true ADK deployment, the environment might manage DB connections.

def setup_cassandra_connection():
    global cluster, session
    try:
        auth_provider = PlainTextAuthProvider(username=CASSANDRA_USER, password=CASSANDRA_PASSWORD)
        cluster = Cluster(contact_points=CASSANDRA_HOSTS, port=CASSANDRA_PORT, auth_provider=auth_provider)
        session = cluster.connect()
        session.set_keyspace(CASSANDRA_KEYSPACE)
        logger.info("Cassandra connection established successfully.")
    except Exception as e:
        logger.error(f"Failed to connect to Cassandra at startup: {e}")
        # Depending on criticality, you might want to exit or log a more severe error

def shutdown_cassandra_connection():
    global cluster, session
    if session:
        session.shutdown()
        session = None
    if cluster:
        cluster.shutdown()
        cluster = None
    logger.info("Cassandra connection shut down.")


