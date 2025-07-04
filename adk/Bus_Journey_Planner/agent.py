from google.adk.agents import SequentialAgent, LlmAgent
from google.adk.models.lite_llm import LiteLlm

# Import the tools from individual agents

import logging
import os

from cassandra.cluster import Cluster
from cassandra.policies import RetryPolicy, ConstantReconnectionPolicy
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from pydantic import BaseModel
import logging
import logging
import json
import requests
from datetime import datetime
from collections import defaultdict
from typing import Optional, Dict
from google.adk.agents import LlmAgent
from google.adk.models import Gemini
from google.adk.tools import FunctionTool
from google.adk.models.lite_llm import LiteLlm
from pydantic import BaseModel
import logging
import os
import json
import urllib.parse
import http.client

import json
import clickhouse_connect
from google.adk.agents import LlmAgent
from google.adk.models import Gemini # This import is not used in the provided code, but kept as is.

from google.adk.models.lite_llm import LiteLlm
from pydantic import BaseModel

from pydantic import BaseModel
import logging
class ReviewSearchResponse(BaseModel):
    status: str
    message: str
    
class RouteSearchResponse(BaseModel):
    status: str
    message: str

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_cassandra_session():
    """Get Cassandra session using environment variables"""
    try:
        contact_points = os.getenv("CASSANDRA_CONTACT_POINTS", "127.0.0.1").split(',')
        port = int(os.getenv("CASSANDRA_PORT", "9042"))
        keyspace = os.getenv("CASSANDRA_KEYSPACE", "ai_bus_info")
        
        logger.info(f"Connecting to Cassandra at {contact_points}:{port}/{keyspace}")
        
        cluster = Cluster(
            contact_points=contact_points,
            port=port,
            connect_timeout=20,
            reconnection_policy=ConstantReconnectionPolicy(delay=3.0, max_attempts=5),
            default_retry_policy=RetryPolicy()
        )
        session = cluster.connect(keyspace)
        logger.info("Connected to Cassandra successfully")
        return session, cluster
    except Exception as e:
        logger.error(f"Error connecting to Cassandra: {e}")
        return None, None

def make_post_request(url, payload, headers):
    try:
        parsed_url = urllib.parse.urlparse(url)
        conn = http.client.HTTPSConnection(parsed_url.netloc) if parsed_url.scheme == 'https' else http.client.HTTPConnection(parsed_url.netloc)
        conn.request("POST", parsed_url.path, json.dumps(payload), headers)
        response = conn.getresponse()
        data = response.read().decode('utf-8')
        conn.close()
        try:
            return json.loads(data)
        except json.JSONDecodeError:
            logger.warning("Invalid JSON response.")
            return None
    except Exception as e:
        logger.error(f"Error making HTTP request to {url}: {e}")
        return None

def generate_nomic_embeddings(text):
    """Generate embeddings using the Nomic API"""
    try:
        api_url = os.getenv("NOMIC_EMBED_API", "http://10.166.8.126:11434/api/embed")
        model_name = os.getenv("EMBEDDING_MODEL", "nomic-embed-text:latest")
        
        logger.info(f"Generating embedding for text: '{text[:50]}...'")
        payload = {"model": model_name, "input": str(text)}
        headers = {'Content-Type': 'application/json'}
        logger.info(f"Making POST request to {api_url} with payload: {payload}")
        result = make_post_request(api_url, payload, headers)
        if result:
            if isinstance(result, str):
                result = json.loads(result)
            embeddings = result.get("embeddings", [])
            if isinstance(embeddings, list) and len(embeddings) > 0:
                embedding = embeddings[0]
                logger.info(f"Successfully generated embedding of length: {len(embedding)}")
                return embedding
        else:
            logger.warning(f"Nomic API request failed.")
            return None
    except Exception as e:
        logger.warning(f"Error in Nomic embedding generation: {e}")
        return None

def perform_ann_query(session_param, table_name, query_embedding, limit=5):
    """Performs an Approximate Nearest Neighbors (ANN) query on the specified Cassandra table."""
    if not session_param:
        logger.error("No valid Cassandra session provided to perform_ann_query.")
        return []

    if not query_embedding:
        logger.error("No query embedding provided for ANN query.")
        return []

    try:
        logger.info(f"Performing ANN query on table {table_name}...")

        query = f"SELECT column_name FROM system_schema.columns WHERE keyspace_name = '{session_param.keyspace}' AND table_name = '{table_name}'"
        rows = session_param.execute(query)
        embedding_col = next((row.column_name for row in rows if row.column_name.endswith('_embedding')), None)

        if not embedding_col:
            logger.error(f"No embedding column found in table {table_name}")
            return []

        query = f"SELECT * FROM {session_param.keyspace}.{table_name} ORDER BY {embedding_col} ANN OF {query_embedding} LIMIT {limit}"
        results = list(session_param.execute(query))
        logger.info(f"ANN query returned {len(results)} results.")
        return results

    except Exception as e:
        logger.error(f"Error performing ANN query: {e}")
        return []

# The ADK Tool function that encapsulates the core logic
def Route_search_tool(user_query: str) -> RouteSearchResponse:
    """
    Fetches the top 5 most relevant bus routes from Cassandra based on a user's query.
    
    Args:
        user_query: The user's query, e.g., "bus from Bangalore to Chennai".
        
    Returns:
        RouteSearchResponse: An object containing the status ('success', 'not_found', 'error') 
                             and a message with the formatted route information or an error detail.
    """
    logger.info(f"Route_search_tool called with query: '{user_query}'")
    try:
        # 1. Generate embedding for the user query
        query_embedding = generate_nomic_embeddings(user_query)
        if not query_embedding:
            return RouteSearchResponse(status="error", message="Could not generate an embedding for the query.")

        # 2. Get Cassandra session
        cassandra_session, _ = get_cassandra_session()
        if not cassandra_session:
            return RouteSearchResponse(status="error", message="Failed to connect to the route database.")
        
        # 3. Perform ANN query
        table_name = os.getenv("CASSANDRA_TABLE", "dpe_route_embedding")
        ann_results = perform_ann_query(cassandra_session, table_name, query_embedding, limit=5)

        # 4. Process and format results
        if not ann_results:
            logger.info(f"No routes found for query: '{user_query}'")
            return RouteSearchResponse(status="not_found", message=f"No routes were found matching your query: '{user_query}'.")
        
        context_info = "Here are the most relevant routes based on your query:\n\n"
        for i, row in enumerate(ann_results):
            context_info += f"--- Route {i+1} ---\n"
            for col_name in row._fields:
                if col_name != 'route_embedding' and hasattr(row, col_name):
                    value = getattr(row, col_name)
                    context_info += f"{col_name.replace('_', ' ').title()}: {value}\n"
            context_info += "\n"
        
        logger.info(f"Successfully found and formatted {len(ann_results)} routes.")
        return "Sandeep is good boy"

    except Exception as e:
        logger.error(f"An unexpected error occurred in Route_search_tool: {e}", exc_info=True)
        return RouteSearchResponse(status="error", message=f"An unexpected error occurred: {str(e)}")
    
def Bus_review_tool(route_id: str) -> ReviewSearchResponse:
    """
    Fetch a single user review for a specific bus route from ClickHouse.
    
    Args:
        route_id: The bus route ID to fetch reviews for. This should be a numeric string.
        
    Returns:
        ReviewSearchResponse: An object containing the status of the operation 
                              ('success', 'not_found', 'error') and a message.
    """
    logger.info(f"Bus_review_tool called with route_id: {route_id}")
    try:
        # Validate route_id is numeric
        if not route_id.isdigit():
            logger.warning(f"Invalid route ID received: '{route_id}'. Must be numeric.")
            return ReviewSearchResponse(
                status="error",
                message=f"Invalid route ID '{route_id}'. Route ID must be numeric."
            )
        
        # Establish connection to ClickHouse
        client = clickhouse_connect.get_client(
            host='10.5.40.193',
            port=8123,
            username='ugc_readonly',
            password='ugc@readonly!',
            secure=False
        )
        
        # SQL query to fetch one review for the given route ID
        # The query uses a subquery to find review IDs associated with the route ID.
        query = """
            SELECT rv.Review
            FROM UGC.UserReviews rv
            WHERE rv.Id IN (
                SELECT r.Id
                FROM UGC.UserRatings r
                WHERE r.RouteID = %(route_id)s
            )
            
        """
        
        logger.debug(f"Executing ClickHouse query for route_id: {route_id}")
        # Execute the query with parameters for safety and clarity
        result = client.query(query, parameters={"route_id": int(route_id)}).result_rows
        
        
            # If a review is found, extract it and return success
            
        logger.info(f"Review found for Route {route_id}. Review text snippet: '{result[:50]}...'")
        return "Karthik is bad boy"
        
            
    except Exception as e:
        # Catch any exceptions during the process and return an error status
        logger.error(f"Error fetching review from ClickHouse for Route {route_id}: {e}", exc_info=True)
        return ReviewSearchResponse(
            status="error",
            message=f"An error occurred while fetching review for Route {route_id}: {str(e)}. Please try again later."
        )

class SeatingSearchResponse(BaseModel):
    """
    Response model for seating search results.
    
    Attributes:
        status: Status of the search operation (success, error, etc.)
        route_id: The bus route ID for which availability was checked
        journey_date: Date of the journey in YYYY-MM-DD format
        available_seats: List of available seats with details
        fare_groups: Dictionary of fare groups with counts and seat numbers
        available_seats_count: Total count of available seats
        total_seats: Total number of seats on the bus
        error: Any error message encountered during the operation
    """
    status: str
    route_id: str
    journey_date: str
    available_seats: list
    fare_groups: dict
    available_seats_count: int
    total_seats: Optional[int] = None
    error: Optional[str] = None

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def get_seat_availability(route_id: str, journey_date: Optional[str] = None) -> SeatingSearchResponse:
    """
    Fetch seat availability for a specific bus route.
    
    Args:
        route_id: The bus route ID to check availability for
        journey_date: Optional date in YYYY-MM-DD format. Defaults to today if not provided.
        
    Returns:
        String containing JSON formatted seat availability data or error message
    """
    try:
        # Validate route_id is numeric
        if not route_id.isdigit():
            return SeatingSearchResponse(
                status="error",
                route_id=route_id,
                journey_date=journey_date or datetime.today().strftime('%Y-%m-%d'),
                available_seats=[],
                fare_groups={},
                available_seats_count=0,
                total_seats=None,
                error=f"Invalid route ID '{route_id}'. Route ID must be numeric."
            )

        # Use today's date if not provided
        if not journey_date:
            journey_date = datetime.today().strftime('%Y-%m-%d')

        api_url = f"http://channels.omega.redbus.in:8001/IASPublic/getRealTimeUpdate/{route_id}/{journey_date}"
        
        all_seat_data = []
        available_seats_count = 0
        fare_availability_groups = defaultdict(lambda: {'count': 0, 'currency': 'INR', 'seat_numbers': []})
        total_seats_from_api = None
        error_message = None

        logger.info(f"Fetching seat availability from: {api_url}")
        response = requests.get(api_url, timeout=15)
        response.raise_for_status()
        data = response.json()

        # Get total seats
        total_seats_from_api = data.get('totalSeats')
        if total_seats_from_api is not None:
            try:
                total_seats_from_api = int(total_seats_from_api)
            except (ValueError, TypeError):
                error_message = f"Warning: 'totalSeats' field invalid: {total_seats_from_api}"
                total_seats_from_api = None
        else:
            error_message = "'totalSeats' field missing in API response."

        # Process seat status
        seat_status_list = data.get('seatStatus', [])
        if not isinstance(seat_status_list, list):
            error_message = "'seatStatus' field is not a list."
            seat_status_list = []
        elif not seat_status_list:
            error_message = "'seatStatus' list is empty."

        for seat_data in seat_status_list:
            if isinstance(seat_data, dict):
                seat_static = seat_data.get('seatStatic', {})
                st_volatile = seat_data.get('stVolatile', {})
                
                if isinstance(seat_static, dict) and isinstance(st_volatile, dict):
                    seat_number = seat_static.get('no') or st_volatile.get('no')
                    seat_availability = st_volatile.get('stAv')
                    fare_info = st_volatile.get('fare', {})
                    seat_type = fare_info.get('seatType')
                    seat_amount = fare_info.get('amount')
                    currency_type = fare_info.get('currencyType', 'INR')

                    seat_info = {
                        'seatNumber': seat_number,
                        'availabilityStatus': seat_availability,
                        'seatType': seat_type,
                        'amount': seat_amount,
                        'currency': currency_type
                    }
                    all_seat_data.append(seat_info)

                    if seat_availability == 'AVAILABLE':
                        available_seats_count += 1
                        if seat_amount is not None:
                            try:
                                fare_key = float(seat_amount)
                                fare_availability_groups[fare_key]['count'] += 1
                                fare_availability_groups[fare_key]['currency'] = currency_type
                                fare_availability_groups[fare_key]['seat_numbers'].append(seat_number)
                            except (ValueError, TypeError):
                                pass

        # # Prepare response
        # return SeatingSearchResponse(
        #     status='success',
        #     route_id=route_id,
        #     journey_date=journey_date,
        #     available_seats=all_seat_data,
        #     fare_groups=dict(fare_availability_groups),
        #     available_seats_count=available_seats_count,
        #     total_seats=total_seats_from_api,
        #     error=error_message
        # )
        return "Sandeep and Karthik both are good boys"

    except requests.exceptions.RequestException as e:
        error_msg = f"API request failed: {str(e)}"
        logger.error(error_msg)
        return SeatingSearchResponse(
            status='error',
            route_id=route_id,
            journey_date=journey_date,
            available_seats=[],
            fare_groups={},
            available_seats_count=0,
            total_seats=None,
            error=error_msg
        )
    
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        logger.error(error_msg)
        return SeatingSearchResponse(
            status='error',
            route_id=route_id,
            journey_date=journey_date,
            available_seats=[],
            fare_groups={},
            available_seats_count=0,
            total_seats=None,
            error=error_msg
        )




Route_agent = LlmAgent(
    name="Route_Agent",
    model=LiteLlm(model="ollama_chat/llama3.2:latest"),
    tools=[Route_search_tool],
    instruction="""
    You are a specialized bus route assistant. Your primary function is to help users find bus routes by using the tools available to you.
    ** Always only give output with the output gotten from the tool, if nothing is generated tell the user that you could not find any routes. Do not ask follow-up questions or re-call the tool for the same request in the same turn.**
    ***Always provide the routes with the route id ,  destination name ,Source name ,highlighted along with other info that you get from the tool's response , nothing else refer to the example below***
    
    **Follow these steps strictly:**
    1.  **Identify User Intent:** Analyze the user's message to understand the route they are asking for (e.g., "from City A to City B").
    2.  **Call the Tool:** Use the `Route_search_tool` with the user's query as the input.
    3.  **Handle Tool Output and Respond:**
        * If the tool returns a `status: "success"`, present the routes from the `message` field clearly and completely to the user. After presenting the information, your task for this turn is finished. Conclude politely.
        * If the tool returns a `status: "not_found"`, inform the user that no matching routes could be found and suggest they rephrase their query. Your task for this turn is then complete.
        * If the tool returns a `status: "error"`, apologize to the user for the technical difficulty, state that you couldn't retrieve the route information, and suggest they try again later. Do not share technical error details. Your task for this turn is then complete.
    4.  **Handle Missing Information:** If the user's message is unclear or does not contain a query for a route, politely ask them to provide the route they are looking for (e.g., "Please tell me the route you're interested in, like 'Delhi to Mumbai'."). Do NOT call the tool without a clear user query.
    
    **Crucially, once you have provided a response based on the tool's output (success, not_found, or error), your current job is done. Do not ask follow-up questions or re-call the tool for the same request in the same turn.**
    For the example output ,
    **Example Output:**
    ```
    --- Route 1 --- 
    Routeid: 14493062 
    Arrtime: 2025-04-17 05:10:00 
    Arrtimezone: UTC+05:30 
    Bustype: A/C Sleeper (2+1) 
    Deptime: 2025-04-16 23:24:00 
    Deptimezone: UTC+05:30 
    Destinationid: 122 
    Destinationname: Bangalore
    Destinationstate: Karnataka 
    Gds: RedBus.Vendor.BitlaNew 
    Isseater: False 
    Issleeper: True 
    Journeydurationinmin: 346 
    Serviceid: 4 
    Servicename: SPS Luxury Travels 
    Slid: 17493470 
    Sourceid: 71390 
    Sourcename: Gobichettipalaiyam 
    Sourcestate: Tamil Nadu 
    Travelsname: SPS Luxury Travels 
    Viacity: ---
    Here the main points are the Destinationname, Source name, Routeid , and if the user as given some date make sure it is the same date,and more importantly if the source and destination matches the user query if not dont output this route. 
    ```
    """,
    output_key="route_information",
)

Review_agent = LlmAgent(
    name="Review_Agent",
    model=LiteLlm(model="ollama_chat/llama3.2:latest"), # Ensure this model is accessible and configured correctly
    tools=[Bus_review_tool], # Register the tool with the agent
    instruction="""
    You are a helpful bus review assistant. Your primary goal is to fetch and present a bus review for a given route ID, and then conclude your response for the current turn.

    **Strict Steps:**
    1.  **Extract Route ID:** Carefully identify and extract the numerical bus route IDs from {route_information}.
    2.  **Call Tool:** Use the `Bus_review_tool` function with the extracted route IDs.
    3.  **Process Tool Result and Respond:**
        * If the tool returns a `status: "success"`, present the review text clearly summarised and politely. After presenting the review, **your task for this turn is complete, and you should provide a final, helpful response to the user.**
        * If the tool returns a `status: "not_found"`, inform the user that no reviews were found for that route ID, and politely offer to check another ID. **Then, your task for this turn is complete.**
        * If the tool returns an `status: "error"`, acknowledge the error gracefully and suggest they try again later or with a different route ID. **Then, your task for this turn is complete.**
    4.  **Handle Missing Route ID:** If the user does not provide a route ID in their initial message, politely ask them to specify one. Do NOT try to call the tool without a route ID.
    **output** output the route information that you got through the output key state along with the reviews as the output in a clear and organised manner.
    **Important:** Once you have provided a response based on the `Bus_review_tool`'s output (whether success, not found, or error), consider the current turn's objective achieved. **Do not re-prompt for the same route ID or re-call the tool for the same request.** Focus on providing a single, conclusive answer for each user query.
    
    """,
    output_key="review_information",
)
Seating_agent = LlmAgent(
    name="Seating_Agent",
    model=LiteLlm(model="ollama_chat/llama3.2:latest"),
    description=("An agent to check bus seat availability for specific routes using RouteID."),
    instruction="""
You are a helpful bus seat availability assistant. You help users check seat availability by:

1. Extracting the route ID from {route_information}
2. Using the get_seat_availability tool to fetch current seat information
3. Presenting the results in a clear, organized format

When showing results, highlight key information like:
- Total available seats
- Different seat types available
- Fare information for different seat categories
- Any relevant warnings or errors

If the route ID is not provided or invalid:
1. Ask the user to provide a valid route ID
2. Explain that the route ID should be a numeric value

For date-specific queries:
1. Accept dates in YYYY-MM-DD format
2. Default to today's date if not specified
3. Validate and explain if the date format is incorrect
    """,
    tools=[get_seat_availability],
    output_key="seat_availability_information",
)
# Review_agent = LlmAgent(
#     name="Review_Agent",
#     model="gemini-2.0-flash", # Ensure this model is accessible and configured correctly
#      # Register the tool with the agent
#     instruction=
#     """
#     just add hi  to {route_information} and then output the review information that you got through the output key and output this 
    
#     """,
#     output_key="review_information",
# )



# Seating_agent = LlmAgent(
#     name="Seating_Agent",
#     model=LiteLlm(model="ollama_chat/llama3.2:latest"),
#     description=("An agent to check bus seat availability for specific routes using RouteID."),
#     instruction="""
# You are a helpful bus seat availability assistant. You help users check seat availability by:

# 1. Extracting the route ID from user queries
# 2. Using the get_seat_availability tool to fetch current seat information
# 3. Presenting the results in a clear, organized format

# When showing results, highlight key information like:
# - Total available seats
# - Different seat types available
# - Fare information for different seat categories
# - Any relevant warnings or errors

# If the route ID is not provided or invalid:
# 1. Ask the user to provide a valid route ID
# 2. Explain that the route ID should be a numeric value

# For date-specific queries:
# 1. Accept dates in YYYY-MM-DD format
# 2. Default to today's date if not specified
# 3. Validate and explain if the date format is incorrect
#     """
    
    
#     ,
#     tools=[get_seat_availability],
#     output_key="seat_availability_information",
# )



# Main Sequential Agent for Complete Journey Planning
Bus_Journey_Planner = SequentialAgent(
    name="Bus_Journey_Planner",
    sub_agents=[Route_agent, Review_agent,Seating_agent],
    description="""
A comprehensive bus journey planning assistant that helps users by:
1. Finding available bus routes between locations
2. Providing user reviews and ratings for the routes
3. Checking seat availability for specific routes
4. Combining all information to help users make informed decisions

The agent follows this workflow:
1. First finds all available routes for the requested journey
2. Then fetches user reviews for the found routes
3. Finally checks seat availability for the selected route
    """
)
root_agent = Bus_Journey_Planner
