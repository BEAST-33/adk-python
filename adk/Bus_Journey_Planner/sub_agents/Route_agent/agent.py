import logging
import os
import json
import urllib.parse
import http.client
import requests
from cassandra.cluster import Cluster
from cassandra.policies import RetryPolicy, ConstantReconnectionPolicy
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Define the response model for the tool
class RouteSearchResponse(BaseModel):
    status: str
    message: str

# Helper functions from the original route_chat_agent.py

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

def perform_ann_query(session_param, table_name, query_embedding, limit=50):
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
        return RouteSearchResponse(status="success", message=context_info)

    except Exception as e:
        logger.error(f"An unexpected error occurred in Route_search_tool: {e}", exc_info=True)
        return RouteSearchResponse(status="error", message=f"An unexpected error occurred: {str(e)}")

# Create the LLM agent
Route_agent = LlmAgent(
    name="Route_Agent",
    model="gemini-2.0-flash",
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
    """
)

# Set the Route_agent as the root agent for standalone execution
root_agent = Route_agent