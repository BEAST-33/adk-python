import os
import logging
import json
import urllib.parse
import http.client
from datetime import datetime
from typing import List, Dict, Optional, Tuple

# ADK imports
from google.adk.agents import LlmAgent
from google.adk.models import Gemini
from google.adk.tools import FunctionTool

import litellm
litellm._turn_on_debug()

# Database and NLP imports
from cassandra.cluster import Cluster, Session
from cassandra.auth import PlainTextAuthProvider
import spacy

# Pydantic imports
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
CASSANDRA_HOSTS = os.environ.get("CASSANDRA_HOSTS", "10.5.10.190").split(",")
CASSANDRA_PORT = 9042
CASSANDRA_USER = "cassandra"
CASSANDRA_PASSWORD = "cassandra"
DEFAULT_KEYSPACE = "ai_bus_info"
DEFAULT_TABLE = "rb_route_embedding"
DEFAULT_EMBEDDING_DIM = 768

NOMIC_API_URL = os.environ.get("NOMIC_API_URL", "http://10.166.8.126:11434/api/embed")
NOMIC_MODEL_NAME = os.environ.get("NOMIC_MODEL_NAME", "nomic-embed-text:latest")

# spaCy models configuration
SPACY_MODELS = {
    'en': 'en_core_web_sm',
    'fr': 'fr_core_news_sm',
    'de': 'de_core_news_sm',
    'es': 'es_core_news_sm',
    'zh': 'zh_core_web_sm',
    'xx': 'xx_ent_wiki_sm',
    'en_lg': 'en_core_web_lg'
}

# Cache loaded models
loaded_models = {}

# Define Pydantic models
class RouteInfo(BaseModel):
    route_id: str
    source: str
    destination: str
    departure_time: str
    arrival_time: str
    bus_type: str
    travels_name: str
    service_name: str
    is_seater: bool
    is_sleeper: bool
    journey_duration_min: int
    similarity_score: float

class RouteSearchResponse(BaseModel):
    status: str
    extracted_source: Optional[str]
    extracted_destination: Optional[str]
    total_results: int
    routes: List[RouteInfo]
    message: Optional[str] = None

def get_nlp(lang_code):
    """Load and cache spaCy models"""
    model_name = SPACY_MODELS.get(lang_code, SPACY_MODELS['en_lg'])
    if model_name not in loaded_models:
        try:
            loaded_models[model_name] = spacy.load(model_name)
            logger.info(f"spaCy model '{model_name}' loaded successfully for language '{lang_code}'.")
        except Exception as e:
            logger.error(f"Failed to load spaCy model '{model_name}': {e}. Falling back to multilingual model.")
            if model_name != SPACY_MODELS['xx']:
                return get_nlp('xx')
            loaded_models[model_name] = None
    return loaded_models[model_name]

# --- Helper Functions ---
def make_post_request(url, payload, headers):
    """Make HTTP POST request for embedding generation"""
    try:
        parsed_url = urllib.parse.urlparse(url)
        conn = http.client.HTTPConnection(parsed_url.hostname, parsed_url.port or 80)
        conn.request("POST", parsed_url.path, json.dumps(payload), headers)
        response = conn.getresponse()
        data = response.read().decode('utf-8')
        conn.close()
        return json.loads(data)
    except Exception as e:
        logger.error(f"Error making HTTP request to {url}: {e}")
        return None

def generate_nomic_embeddings(text, api_url=NOMIC_API_URL, model_name=NOMIC_MODEL_NAME):
    """Generate embeddings using Nomic API"""
    payload = {"model": model_name, "input": str(text)}
    headers = {'Content-Type': 'application/json'}
    result = make_post_request(api_url, payload, headers)
    if not result:
        return None
    embeddings = result.get("embeddings", [])
    if isinstance(embeddings, list) and len(embeddings) > 0:
        return embeddings[0]
    return None

def get_session() -> Optional[Session]:
    """Get Cassandra session"""
    try:
        auth_provider = PlainTextAuthProvider(
            username=CASSANDRA_USER, password=CASSANDRA_PASSWORD
        )
        cluster = Cluster(
            contact_points=CASSANDRA_HOSTS, port=CASSANDRA_PORT, auth_provider=auth_provider
        )
        session = cluster.connect()
        return session
    except Exception as e:
        logger.error(f"Failed to connect to Cassandra: {e}")
        return None

def extract_source_destination(text: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract source and destination from text using spaCy NER"""
    nlp = get_nlp("en_lg")
    if nlp is None:
        logger.warning("spaCy model not loaded, cannot extract source/destination.")
        return None, None

    doc = nlp(text)
    source = None
    destination = None

    # Strategy 1: Look for "from X to Y" patterns
    try:
        tokens = [token.lower_ for token in doc]
        from_index = tokens.index("from")
        
        for i in range(from_index + 1, len(tokens)):
            if tokens[i] == "to":
                to_index = i
                
                # Extract source (between "from" and "to")
                source_candidate_tokens = []
                for j in range(from_index + 1, to_index):
                    if doc[j].ent_type_ in ["GPE", "LOC"]:
                        source_candidate_tokens.append(doc[j].text)
                if source_candidate_tokens:
                    source = " ".join(source_candidate_tokens).title()

                # Extract destination (after "to")
                destination_candidate_tokens = []
                for k in range(to_index + 1, len(tokens)):
                    if doc[k].ent_type_ in ["GPE", "LOC"]:
                        destination_candidate_tokens.append(doc[k].text)
                if destination_candidate_tokens:
                    destination = " ".join(destination_candidate_tokens).title()
                
                if source and destination:
                    break
    except ValueError:
        pass

    # Strategy 2: Fallback to general GPE/LOC entities
    if not source or not destination:
        potential_locations = []
        for ent in doc.ents:
            if ent.label_ in ["GPE", "LOC"]:
                potential_locations.append(ent.text)

        if not source and potential_locations:
            source = potential_locations[0].title()
        if not destination and len(potential_locations) > 1:
            for loc in potential_locations:
                if loc.title() != source:
                    destination = loc.title()
                    break

    logger.info(f"Extracted Source: '{source}', Destination: '{destination}' from text: '{text}'")
    return source, destination

def get_time_from_datetime_str(dt_input):
    """Convert datetime string to time format"""
    if isinstance(dt_input, datetime):
        return dt_input.strftime("%H:%M")
    
    if dt_input and dt_input != 'N/A':
        try:
            dt_object = datetime.strptime(dt_input, "%Y-%m-%d %H:%M:%S")
            return dt_object.strftime("%H:%M")
        except ValueError:
            return 'N/A'
    return 'N/A'

def route_search_function(query: str, top_k: int = 5) -> str:
    """
    Search for bus routes using natural language query.
    
    Args:
        query: Natural language query describing the bus route
        top_k: Number of top results to return
        
    Returns:
        String containing JSON formatted search results or error message
    """
    logger.info(f"Starting route search with query: {query}")
    try:
        # Extract source and destination
        source, destination = extract_source_destination(query)
        logger.info(f"Extracted source: {source}, destination: {destination}")
        
        # Generate embedding for the query
        search_text = f"{source} to {destination}" if source and destination else query
        logger.info(f"Generating embedding for: {search_text}")
        embedding = generate_nomic_embeddings(search_text)
        
        if not embedding:
            logger.error("Failed to generate embeddings")
            error_response = RouteSearchResponse(
                status="error",
                extracted_source=source,
                extracted_destination=destination,
                total_results=0,
                routes=[],
                message="Failed to generate embeddings for the query."
            )
            return error_response.model_dump_json()
        
        # Connect to Cassandra and perform ANN search
        logger.info("Attempting to connect to Cassandra")
        session = get_session()
        if not session:
            logger.error("Failed to establish Cassandra session")
            error_response = RouteSearchResponse(
                status="error",
                extracted_source=source,
                extracted_destination=destination,
                total_results=0,
                routes=[],
                message="Failed to connect to Cassandra database."
            )
            return error_response.model_dump_json()
        
        try:
            # ANN Query
            logger.info("Executing ANN query")
            ann_query = f"""
            SELECT routeid, arrtime, bustype, deptime, destinationid, destinationname,
                   destinationstate, isseater, issleeper, journeydurationinmin,
                   serviceid, servicename, slid, sourceid, sourcename, sourcestate,
                   travelsname, similarity_cosine(route_embedding, ?) AS cosine_similarity
            FROM {DEFAULT_KEYSPACE}.{DEFAULT_TABLE}
            ORDER BY route_embedding ANN OF ?
            LIMIT ?
            """
            
            prepared = session.prepare(ann_query)
            rows = session.execute(prepared, [embedding, embedding, top_k])
            
            # Process results
            results = []
            for row in rows:
                logger.info(f"Processing route: {getattr(row, 'routeid', 'N/A')}")
                
                route_info = {
                    "route_id": str(getattr(row, 'routeid', 'N/A')),  # Convert to string
                    "source": getattr(row, 'sourcename', 'N/A'),
                    "destination": getattr(row, 'destinationname', 'N/A'),
                    "departure_time": get_time_from_datetime_str(getattr(row, 'deptime', 'N/A')),
                    "arrival_time": get_time_from_datetime_str(getattr(row, 'arrtime', 'N/A')),
                    "bus_type": getattr(row, 'bustype', 'N/A'),
                    "travels_name": getattr(row, 'travelsname', 'N/A'),
                    "service_name": getattr(row, 'servicename', 'N/A'),
                    "is_seater": getattr(row, 'isseater', False),
                    "is_sleeper": getattr(row, 'issleeper', False),
                    "journey_duration_min": getattr(row, 'journeydurationinmin', 0),
                    "similarity_score": float(getattr(row, 'cosine_similarity', 0.0))
                }
                results.append(route_info)
            
            logger.info(f"Found {len(results)} matching routes")
            
            # Prepare response
            response = RouteSearchResponse(
                status="success",
                extracted_source=source,
                extracted_destination=destination,
                total_results=len(results),
                routes=[RouteInfo(**route) for route in results]
            )
            logger.info(f"Route search completed successfully")
            return response.model_dump_json()
            
        finally:
            logger.info("Closing Cassandra session")
            session.cluster.shutdown()
            
    except Exception as e:
        logger.error(f"Error in route search: {str(e)}", exc_info=True)
        error_response = RouteSearchResponse(
            status="error",
            extracted_source=None,
            extracted_destination=None,
            total_results=0,
            routes=[],
            message=str(e)
        )
        return error_response.model_dump_json()

# Note: For ADK Agent class, we pass functions directly to tools, not wrapped in FunctionTool

# Create the ADK Agent (not LlmAgent directly)
from google.adk.agents import Agent

root_agent = Agent(
    name="Route_agent",
    model=LiteLlm(model="ollama_chat/llama3"),  
    description="Agent to search for bus routes between cities",
    instruction="""
You are a helpful bus route search assistant. You help users find bus routes by:

1. Understanding their natural language queries about bus travel
2. Using the route_search tool to find relevant bus routes
3. Presenting the results in a clear, user-friendly format

When a user asks about bus routes, always use the route_search tool to find matching routes.
Present the results in an organized way, highlighting key information like:
- Route details (source to destination)
- Departure and arrival times
- Bus type and amenities (seater/sleeper)
- Travel company
- Journey duration

If no routes are found, suggest alternative approaches or ask for more specific information.
Always provide helpful and detailed responses based on the search results.
    """,
    tools=[route_search_function]  
)

if __name__ == "__main__":
    print("Route Agent initialized successfully!")
    print("To interact with the agent, use one of these methods:")
    print("1. Run 'adk run route_agent' from the parent directory")
    print("2. Run 'adk web' from the parent directory for web UI")
    print("3. Test the route_search function directly:")
    
    # Test the function directly
    test_queries = [
        "I need a bus from Delhi to Mumbai",
        "Find overnight buses from Bangalore to Chennai", 
        "Show me routes from Kolkata to Pune"
    ]
    
    for query in test_queries:
        print(f"\n--- Testing Query: {query} ---")
        try:
            result = route_search_function(query)
            print(f"Result: {result}")
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()