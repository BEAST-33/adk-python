# bus_review_rating_route_mcp_server.py
import asyncio
import json
import os
from dotenv import load_dotenv
import logging
import clickhouse_connect
from pydantic import BaseModel
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import urllib.parse
import http.client
import spacy

# Cassandra imports
from cassandra.cluster import Cluster, Session
from cassandra.auth import PlainTextAuthProvider

# MCP Server Imports
from mcp import types as mcp_types # Use alias to avoid conflict
from mcp.server.lowlevel import Server, NotificationOptions
from mcp.server.models import InitializationOptions
import mcp.server.stdio # For running as a stdio server

# ADK Tool Imports
from google.adk.tools.function_tool import FunctionTool
# ADK <-> MCP Conversion Utility
from google.adk.tools.mcp_tool.conversion_utils import adk_to_mcp_tool_type

# --- Load Environment Variables (If ADK tools need them, e.g., API keys) ---
load_dotenv() # Create a .env file in the same directory if needed

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Define the response model for the review tool
class ReviewSearchResponse(BaseModel):
    status: str
    message: str

# Define the response model for the rating tool
class BusRatingResponse(BaseModel):
    status: str
    message: str
    route_id: str
    average_rating: float | None = None

# Define the response model for the route agent tool
class RouteAgentResponse(BaseModel):
    status: str
    message: str
    source_extracted: Optional[str] = None
    destination_extracted: Optional[str] = None
    matches: Optional[List[Dict]] = None

# --- Configuration for Route Agent ---
CASSANDRA_HOSTS = os.environ.get("CASSANDRA_HOSTS", "10.5.10.190").split(",")
CASSANDRA_PORT = 9042
CASSANDRA_USER = "cassandra"
CASSANDRA_PASSWORD = "cassandra"
DEFAULT_KEYSPACE = "ai_bus_info"
DEFAULT_TABLE = "rb_route_embedding"
DEFAULT_EMBEDDING_DIM = 768

NOMIC_API_URL = os.environ.get("NOMIC_API_URL", "http://10.166.8.126:11434/api/embed")
NOMIC_MODEL_NAME = os.environ.get("NOMIC_MODEL_NAME", "nomic-embed-text:latest")

# Map language codes to spaCy model names
SPACY_MODELS = dict(en='en_core_web_sm', fr='fr_core_news_sm', de='de_core_news_sm', es='es_core_news_sm',
                    zh='zh_core_web_sm', xx='xx_ent_wiki_sm', en_lg ='en_core_web_lg')

# Cache loaded models
loaded_models = {}

def get_nlp(lang_code):
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

# --- Helper: Nomic Embedding ---
def make_post_request(url, payload, headers):
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
    payload = {"model": model_name, "input": str(text)}
    headers = {'Content-Type': 'application/json'}
    result = make_post_request(api_url, payload, headers)
    if not result:
        return None
    embeddings = result.get("embeddings", [])
    if isinstance(embeddings, list) and len(embeddings) > 0:
        return embeddings[0]
    return None

# --- Helper: Cassandra ANN ---
def get_session() -> Optional[Session]:
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

def get_table_columns(session: Session, keyspace: str, table: str) -> Optional[List[str]]:
    try:
        logger.info(f"About to query system_schema.columns for keyspace='{keyspace}', table='{table}'")
        rows = session.execute(
            "SELECT column_name FROM system_schema.columns WHERE keyspace_name=%s AND table_name=%s",
            (keyspace, table),
        )
        rows_list = list(rows)
        logger.info(f"Raw rows received from system_schema.columns: {rows_list}")
        return [row.column_name for row in rows_list]
    except Exception as e:
        logger.error(f"Failed to retrieve columns for {keyspace}.{table}: {e}")
        return None

def get_embedding_column(columns: List[str]) -> Optional[str]:
    for column in columns:
        if "embedding" in column.lower():
            return column
    return None

def process_rows(rows, columns, target_sourcename=None, target_destinationname=None):
    results = []
    for row in rows:
        current_sourcename = row.sourcename if hasattr(row, 'sourcename') else None
        current_destinationname = row.destinationname if hasattr(row, 'destinationname') else None

        if target_sourcename is not None:
            if current_sourcename is None or current_sourcename != target_sourcename:
                logger.debug(f"Row skipped: Sourcename '{current_sourcename}' doesn't match '{target_sourcename}'")
                continue

        if target_destinationname is not None:
            if current_destinationname is None or current_destinationname != target_destinationname:
                logger.debug(f"Row skipped: Destinationname '{current_destinationname}' doesn't match '{target_destinationname}'")
                continue

        results.append(row)

    logger.info(f"Total matches found after filtering: {len(results)}")
    return results

def ann_query_on_cassandra(
        session: Session,
        keyspace: str,
        table: str,
        embedding_column: str,
        embedding: List[float],
        dim: int,
        source_name: Optional[str] = None,
        destination_name: Optional[str] = None,
        top_k: int = 50,
) -> Optional[List[Dict]]:
    try:
        query_parts = [
            f"""SELECT
    routeid,
    arrtime,
    bustype,
    deptime,
    destinationid,
    destinationname,
    destinationstate,
    isseater,
    issleeper,
    journeydurationinmin,
    serviceid,
    servicename,
    slid,
    sourceid,
    sourcename,
    sourcestate,
    travelsname,
    similarity_cosine({embedding_column}, ?) AS cosine_similarity"""
        ]
        query_parts.append(f"FROM {keyspace}.{table}")

        where_clauses = []
        bound_params = [embedding]

        if where_clauses:
            query_parts.append("WHERE " + " AND ".join(where_clauses))

        query_parts.append(f"ORDER BY {embedding_column} ANN OF ?")
        bound_params.append(embedding)

        query_parts.append(f"LIMIT {top_k}")

        query = " ".join(query_parts)

        logger.info(f"Preparing ANN query: {query}")
        logger.info(f"Bound parameters (excluding embedding values): {bound_params[1:]}")

        prepared = session.prepare(query)
        logger.info("Executing ANN query on Cassandra...")
        rows = session.execute(prepared, bound_params)

        columns = [
            'routeid',
            'arrtime',
            'arrtimezone',
            'bustype',
            'deptime',
            'deptimezone',
            'destinationid',
            'destinationname',
            'destinationstate',
            'gds',
            'isseater',
            'issleeper',
            'journeydurationinmin',
            'route_embedding',
            'serviceid',
            'servicename',
            'slid',
            'sourceid',
            'sourcename',
            'sourcestate',
            'travelsname',
            'viacity',
            'cosine_similarity'
        ]

        results = process_rows(rows, columns, source_name, destination_name)
        return results
    except Exception as e:
        logger.error(f"Error during ANN query on {keyspace}.{table}: {e}")
        return None

def extract_source_destination(text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extracts potential source and destination names from free text using spaCy.
    This version tries to prioritize explicit "from X to Y" patterns and then falls back to NER.
    """
    nlp = get_nlp("en_core_web_lg")
    if nlp is None:
        logger.warning("spaCy model not loaded, cannot extract source/destination.")
        return None, None

    doc = nlp(text)
    source = None
    destination = None

    text_lower = text.lower()

    # Simple token-based search for "from ... to ..."
    try:
        tokens = [token.lower_ for token in doc]

        from_index = tokens.index("from")
        for i in range(from_index + 1, len(tokens)):
            if tokens[i] == "to":
                to_index = i

                source_candidate_tokens = []
                for j in range(from_index + 1, to_index):
                    if doc[j].ent_type_ in ["GPE", "LOC"]:
                        source_candidate_tokens.append(doc[j].text)
                if source_candidate_tokens:
                    source = " ".join(source_candidate_tokens).title()

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

    if not source and not destination and " to " in text_lower:
        parts = text_lower.split(" to ")
        if len(parts) >= 2:
            last_to_idx = text_lower.rfind(" to ")
            if last_to_idx != -1:
                after_to_text = text[last_to_idx + len(" to "):]
                doc_after_to = nlp(after_to_text)
                for ent in doc_after_to.ents:
                    if ent.label_ in ["GPE", "LOC"]:
                        destination = ent.text.title()
                        break

            first_to_idx = text_lower.find(" to ")
            if first_to_idx != -1:
                before_to_text = text[:first_to_idx]
                doc_before_to = nlp(before_to_text)
                for ent in reversed(doc_before_to.ents):
                    if ent.label_ in ["GPE", "LOC"]:
                        source = ent.text.title()
                        break

    potential_locations_ner = []
    if not source or not destination:
        for ent in doc.ents:
            if ent.label_ in ["GPE", "LOC"]:
                potential_locations_ner.append(ent.text)

        if not source and potential_locations_ner:
            source = potential_locations_ner[0].title()
        if not destination and len(potential_locations_ner) > 1:
            for loc in potential_locations_ner:
                if loc.title() != source:
                    destination = loc.title()
                    break
            if not destination and len(potential_locations_ner) == 1 and not source:
                destination = potential_locations_ner[0].title()

    logger.info(f"Extracted Source: '{source}', Destination: '{destination}' from text: '{text}'")
    return source, destination

def get_time_from_datetime_str(dt_input):
    if isinstance(dt_input, datetime):
        return dt_input.strftime("%H:%M")
    
    if dt_input and dt_input != 'N/A':
        try:
            dt_object = datetime.strptime(dt_input, "%Y-%m-%d %H:%M:%S")
            return dt_object.strftime("%H:%M")
        except ValueError:
            return 'N/A'
    return 'N/A'

# --- ClickHouse Client Helper ---
def get_client(host, port, username, password, secure):
    """Establishes and returns a ClickHouse client connection."""
    return clickhouse_connect.get_client(
        host=host,
        port=port,
        username=username,
        password=password,
        secure=secure
    )

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
        if not route_id.isdigit():
            logger.warning(f"Invalid route ID received: '{route_id}'. Must be numeric.")
            return ReviewSearchResponse(
                status="error",
                message=f"Invalid route ID '{route_id}'. Route ID must be numeric."
            )
        
        client = get_client(
            host='10.5.40.193',
            port=8123,
            username='ugc_readonly',
            password='ugc@readonly!',
            secure=False
        )
        
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
        result = client.query(query, parameters={"route_id": int(route_id)}).result_rows
        
        if result:
            reviews = []
            for row in result:
                if row and row[0]:
                    reviews.append(str(row[0]).strip())
            
            if reviews:
                if len(reviews) == 1:
                    review_text = reviews[0]
                else:
                    review_text = "\n\n".join([f"Review {i+1}: {review}" for i, review in enumerate(reviews)])
                
                logger.info(f"Found {len(reviews)} review(s) for Route {route_id}")
                return ReviewSearchResponse(
                    status="success",
                    message=f"Reviews for Route {route_id}:\n\n{review_text}"
                )
            else:
                logger.info(f"No valid review content found for Route {route_id}")
                return ReviewSearchResponse(
                    status="not_found",
                    message=f"No valid review content found for Route {route_id}"
                )
        else:
            logger.info(f"No reviews found for Route {route_id}")
            return ReviewSearchResponse(
                status="not_found",
                message=f"No reviews found for Route {route_id}"
            )
            
    except Exception as e:
        logger.error(f"Error fetching review from ClickHouse for Route {route_id}: {e}", exc_info=True)
        return ReviewSearchResponse(
            status="error",
            message=f"An error occurred while fetching review for Route {route_id}: {str(e)}. Please try again later."
        )

def get_bus_rating_tool(route_id: str) -> BusRatingResponse:
    """
    Fetches the average user rating for a specific bus route from ClickHouse.

    Args:
        route_id: The bus route ID to fetch the rating for. This should be a numeric string.

    Returns:
        BusRatingResponse: An object containing the status, message, route_id, and average_rating.
    """
    logger.info(f"get_bus_rating_tool called with route_id: {route_id}")
    try:
        if not route_id.isdigit():
            logger.warning(f"Invalid route ID received: '{route_id}'. Must be numeric.")
            return BusRatingResponse(
                status="error",
                message=f"Invalid route ID '{route_id}'. Route ID must be numeric.",
                route_id=route_id,
                average_rating=None
            )

        client = get_client(
            host='10.5.40.193',
            port=8123,
            username='ugc_readonly',
            password='ugc@readonly!',
            secure=False
        )
        query = "SELECT Value FROM UGC.UserRatings WHERE RouteID = %(route_id)s"
        logger.debug(f"Executing ClickHouse query for route_id: {route_id}")
        result = client.query(query, parameters={"route_id": int(route_id)})
        rows = result.result_rows

        if rows:
            ratings = [row[0] for row in rows]
            average_rating = round(sum(ratings) / len(ratings), 1)
            logger.info(f"Average bus score for RouteID {route_id}: {average_rating}")
            return BusRatingResponse(
                status="success",
                message=f"Average bus rating fetched for RouteID {route_id}.",
                route_id=route_id,
                average_rating=average_rating
            )
        else:
            logger.info(f"No data found for RouteID {route_id}")
            return BusRatingResponse(
                status="not_found",
                message=f"No ratings found for RouteID {route_id}.",
                route_id=route_id,
                average_rating=None
            )

    except Exception as e:
        logger.error(f"Error fetching bus rating from ClickHouse for RouteID {route_id}: {e}", exc_info=True)
        return BusRatingResponse(
            status="error",
            message=f"An error occurred while fetching rating for RouteID {route_id}: {str(e)}. Please try again later.",
            route_id=route_id,
            average_rating=None
        )

def route_search_tool(
    user_text: str,
    keyspace: str = DEFAULT_KEYSPACE,
    table: str = DEFAULT_TABLE,
    dimension: int = DEFAULT_EMBEDDING_DIM
) -> RouteAgentResponse:
    """
    Route search tool that accepts text, extracts potential source and destination,
    generates a Nomic embedding, and runs ANN queries on a Cassandra database.
    
    Args:
        user_text: The user input text to search for bus routes
        keyspace: Cassandra keyspace to search in
        table: Cassandra table to search in
        dimension: Expected embedding dimension
        
    Returns:
        RouteAgentResponse: An object containing the status, message, extracted locations, and matches
    """
    logger.info(f"route_search_tool called with user_text: '{user_text}'")
    
    try:
        if not user_text.strip():
            return RouteAgentResponse(
                status="error",
                message="No text provided for route search."
            )

        # Extract source and destination from user text
        extracted_source, extracted_destination = extract_source_destination(user_text)

        # Generate embedding
        embedding_text = f"{extracted_source or ''} to {extracted_destination or ''}".strip()
        if not embedding_text or embedding_text == "to":
            embedding_text = user_text
            
        embedding = generate_nomic_embeddings(embedding_text)
        if not embedding:
            return RouteAgentResponse(
                status="error",
                message="Could not generate embedding for the provided text."
            )
            
        if len(embedding) != dimension:
            return RouteAgentResponse(
                status="error",
                message=f"Embedding dimension mismatch: expected {dimension}, got {len(embedding)}"
            )

        # Connect to Cassandra and perform ANN query
        session = get_session()
        if not session:
            return RouteAgentResponse(
                status="error",
                message="Failed to connect to Cassandra database."
            )

        try:
            columns = get_table_columns(session, keyspace, table)
            if not columns:
                return RouteAgentResponse(
                    status="error",
                    message=f"Could not retrieve columns for table '{table}' in keyspace '{keyspace}'."
                )
                
            embedding_column = get_embedding_column(columns)
            if not embedding_column:
                return RouteAgentResponse(
                    status="error",
                    message=f"No suitable embedding column found in table '{table}'."
                )

            # Perform ANN query
            matches = ann_query_on_cassandra(
                session,
                keyspace,
                table,
                embedding_column,
                embedding,
                dimension,
                source_name=extracted_source,
                destination_name=extracted_destination,
                top_k=50
            )

            if matches is not None:
                # Convert matches to dictionary format for JSON serialization
                matches_dict = []
                for match in matches:
                    match_dict = {
                        'routeid': getattr(match, 'routeid', 'N/A'),
                        'arrtime': get_time_from_datetime_str(getattr(match, 'arrtime', 'N/A')),
                        'bustype': getattr(match, 'bustype', 'N/A'),
                        'deptime': get_time_from_datetime_str(getattr(match, 'deptime', 'N/A')),
                        'destinationid': getattr(match, 'destinationid', 'N/A'),
                        'destinationname': getattr(match, 'destinationname', 'N/A'),
                        'destinationstate': getattr(match, 'destinationstate', 'N/A'),
                        'isseater': getattr(match, 'isseater', 'N/A'),
                        'issleeper': getattr(match, 'issleeper', 'N/A'),
                        'journeydurationinmin': getattr(match, 'journeydurationinmin', 'N/A'),
                        'serviceid': getattr(match, 'serviceid', 'N/A'),
                        'servicename': getattr(match, 'servicename', 'N/A'),
                        'slid': getattr(match, 'slid', 'N/A'),
                        'sourceid': getattr(match, 'sourceid', 'N/A'),
                        'sourcename': getattr(match, 'sourcename', 'N/A'),
                        'sourcestate': getattr(match, 'sourcestate', 'N/A'),
                        'travelsname': getattr(match, 'travelsname', 'N/A'),
                        'cosine_similarity': getattr(match, 'cosine_similarity', 0)
                    }
                    matches_dict.append(match_dict)

                status_desc = f"Found {len(matches_dict)} matching bus routes."
                if extracted_source or extracted_destination:
                    status_desc += " Results filtered by detected locations."
                    
                return RouteAgentResponse(
                    status="success",
                    message=status_desc,
                    source_extracted=extracted_source,
                    destination_extracted=extracted_destination,
                    matches=matches_dict
                )
            else:
                return RouteAgentResponse(
                    status="error",
                    message="Error occurred during route search query."
                )
                
        finally:
            if session:
                session.cluster.shutdown()

    except Exception as e:
        logger.error(f"Error in route_search_tool: {e}", exc_info=True)
        return RouteAgentResponse(
            status="error",
            message=f"An error occurred during route search: {str(e)}. Please try again later."
        )

# --- Prepare the ADK Tools ---
print("Initializing ADK Bus_review_tool...")
adk_review_tool = FunctionTool(Bus_review_tool)
print(f"ADK tool '{adk_review_tool.name}' initialized and ready to be exposed via MCP.")

print("Initializing ADK get_bus_rating_tool...")
adk_rating_tool = FunctionTool(get_bus_rating_tool)
print(f"ADK tool '{adk_rating_tool.name}' initialized and ready to be exposed via MCP.")

print("Initializing ADK route_search_tool...")
adk_route_tool = FunctionTool(route_search_tool)
print(f"ADK tool '{adk_route_tool.name}' initialized and ready to be exposed via MCP.")

# Store tools in a dictionary for easy lookup
adk_tools = {
    adk_review_tool.name: adk_review_tool,
    adk_rating_tool.name: adk_rating_tool,
    adk_route_tool.name: adk_route_tool
}

# --- MCP Server Setup ---
print("Creating MCP Server instance...")
app = Server("bus-review-rating-route-mcp-server")

# Implement the MCP server's handler to list available tools
@app.list_tools()
async def list_mcp_tools() -> list[mcp_types.Tool]:
    """MCP handler to list tools this server exposes."""
    print("MCP Server: Received list_tools request.")
    mcp_tools = []
    for tool_name, adk_tool in adk_tools.items():
        mcp_tool_schema = adk_to_mcp_tool_type(adk_tool)
        mcp_tools.append(mcp_tool_schema)
        print(f"MCP Server: Advertising tool: {mcp_tool_schema.name}")
    return mcp_tools

# Implement the MCP server's handler to execute a tool call
@app.call_tool()
async def call_mcp_tool(
    name: str, arguments: dict
) -> list[mcp_types.Content]:
    """MCP handler to execute a tool call requested by an MCP client."""
    print(f"MCP Server: Received call_tool request for '{name}' with args: {arguments}")

    if name in adk_tools:
        try:
            adk_tool = adk_tools[name]
            
            adk_tool_response = await adk_tool.run_async(
                args=arguments,
                tool_context=None,
            )
            print(f"MCP Server: ADK tool '{name}' executed. Response: {adk_tool_response}")

            # Handle Pydantic model response - convert to dict if needed
            if hasattr(adk_tool_response, 'model_dump'):
                response_dict = adk_tool_response.model_dump()
            elif hasattr(adk_tool_response, 'dict'):
                response_dict = adk_tool_response.dict()
            else:
                response_dict = adk_tool_response

            response_text = json.dumps(response_dict, indent=2)
            return [mcp_types.TextContent(type="text", text=response_text)]

        except Exception as e:
            print(f"MCP Server: Error executing ADK tool '{name}': {e}")
            error_text = json.dumps({"error": f"Failed to execute tool '{name}': {str(e)}"})
            return [mcp_types.TextContent(type="text", text=error_text)]
    else:
        print(f"MCP Server: Tool '{name}' not found/exposed by this server.")
        error_text = json.dumps({"error": f"Tool '{name}' not implemented by this server."})
        return [mcp_types.TextContent(type="text", text=error_text)]

# --- MCP Server Runner ---
async def run_mcp_stdio_server():
    """Runs the MCP server, listening for connections over standard input/output."""
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        print("MCP Stdio Server: Starting handshake with client...")
        await app.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name=app.name,
                server_version="0.1.0",
                capabilities=app.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )
        print("MCP Stdio Server: Run loop finished or client disconnected.")

if __name__ == "__main__":
    print("Launching MCP Server to expose ADK Bus Review, Rating, and Route Search tools via stdio...")
    try:
        asyncio.run(run_mcp_stdio_server())
    except KeyboardInterrupt:
        print("\nMCP Server (stdio) stopped by user.")
    except Exception as e:
        print(f"MCP Server (stdio) encountered an error: {e}")
    finally:
        print("MCP Server (stdio) process exiting.")
