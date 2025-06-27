import os
import logging
import json
import urllib.parse
import http.client
from datetime import datetime
from typing import List, Dict, Optional, Tuple

from cassandra.cluster import Cluster, Session
from cassandra.auth import PlainTextAuthProvider

import spacy
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.DEBUG)
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

# Map language codes to spaCy model names
SPACY_MODELS = dict(en='en_core_web_sm', fr='fr_core_news_sm', de='de_core_news_sm', es='es_core_news_sm',
                    zh='zh_core_web_sm', xx='xx_ent_wiki_sm', en_lg ='en_core_web_lg')

# Cache loaded models
loaded_models = {}

# Define the response model for the tool
class RouteSearchResponse(BaseModel):
    status: str
    message: str
    source_extracted: Optional[str] = None
    destination_extracted: Optional[str] = None
    matches: Optional[List[Dict]] = None

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
        logging.error(f"Error making HTTP request to {url}: {e}")
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
        logging.error(f"Failed to connect to Cassandra: {e}")
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
    """
    Processes a list of Cassandra rows, directly accessing column values
    using dot notation (e.g., row.column_name) and applying filters
    sequentially based on sourcename and then destinationname.
    """
    results = []
    for row in rows:
        current_sourcename = row.sourcename if hasattr(row, 'sourcename') else None
        current_destinationname = row.destinationname if hasattr(row, 'destinationname') else None

        # Step 1: Filter by sourcename (if provided)
        if target_sourcename is not None:
            if current_sourcename is None or current_sourcename != target_sourcename:
                logger.debug(f"Row skipped: Sourcename '{current_sourcename}' doesn't match '{target_sourcename}'")
                continue

        # Step 2: Filter by destinationname (if provided)
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
            'routeid', 'arrtime', 'arrtimezone', 'bustype', 'deptime', 'deptimezone',
            'destinationid', 'destinationname', 'destinationstate', 'gds', 'isseater',
            'issleeper', 'journeydurationinmin', 'route_embedding', 'serviceid',
            'servicename', 'slid', 'sourceid', 'sourcename', 'sourcestate',
            'travelsname', 'viacity', 'cosine_similarity'
        ]

        results = process_rows(rows, columns, source_name, destination_name)
        return results
    except Exception as e:
        logger.error(f"Error during ANN query on {keyspace}.{table}: {e}")
        return None

def extract_source_destination(text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extracts potential source and destination names from free text using spaCy.
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

    # If "from X to Y" didn't yield results, try "X to Y" directly
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

    # Fallback to general GPE/LOC entities if patterns fail
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

def Bus_route_search_tool(user_text: str, keyspace: str = DEFAULT_KEYSPACE, table: str = DEFAULT_TABLE, dimension: int = DEFAULT_EMBEDDING_DIM) -> RouteSearchResponse:
    """
    Search for bus routes based on user text input using ANN search on Cassandra.
    
    Args:
        user_text: The user's text input containing route search criteria
        keyspace: Cassandra keyspace to search in
        table: Cassandra table to search in
        dimension: Expected embedding dimension
        
    Returns:
        RouteSearchResponse: An object containing the status, message, and search results
    """
    logger.info(f"Bus_route_search_tool called with user_text: {user_text}")
    
    try:
        if not user_text:
            return RouteSearchResponse(
                status="error",
                message="No text provided for route search."
            )

        # Extract source and destination from user text
        extracted_source, extracted_destination = extract_source_destination(user_text)

        # Generate embedding
        embedding_text = f"{extracted_source or ''} to {extracted_destination or ''}".strip()
        if embedding_text == "to":
            embedding_text = user_text
            
        embedding = generate_nomic_embeddings(embedding_text)
        if not embedding:
            return RouteSearchResponse(
                status="error",
                message="Could not generate embedding for the provided text."
            )
            
        if len(embedding) != dimension:
            return RouteSearchResponse(
                status="error",
                message=f"Embedding dimension mismatch: expected {dimension}, got {len(embedding)}"
            )

        # Connect to Cassandra and perform ANN query
        session = get_session()
        if not session:
            return RouteSearchResponse(
                status="error",
                message="Failed to connect to Cassandra database."
            )

        try:
            columns = get_table_columns(session, keyspace, table)
            if not columns:
                return RouteSearchResponse(
                    status="error",
                    message=f"Could not retrieve columns for table '{table}' in keyspace '{keyspace}'."
                )
                
            embedding_column = get_embedding_column(columns)
            if not embedding_column:
                return RouteSearchResponse(
                    status="error",
                    message=f"No suitable embedding column found in table '{table}'."
                )

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

            if matches is not None and len(matches) > 0:
                # Convert matches to serializable format and limit to top 5
                serializable_matches = []
                top_5_matches = matches[:5]  # Ensure we only take top 5
                
                for i, m in enumerate(top_5_matches, 1):
                    match_dict = {
                        'rank': i,
                        'routeid': getattr(m, 'routeid', 'N/A'),
                        'sourcename': getattr(m, 'sourcename', 'N/A'),
                        'sourceid': getattr(m, 'sourceid', 'N/A'),
                        'sourcestate': getattr(m, 'sourcestate', 'N/A'),
                        'destinationname': getattr(m, 'destinationname', 'N/A'),
                        'destinationid': getattr(m, 'destinationid', 'N/A'),
                        'destinationstate': getattr(m, 'destinationstate', 'N/A'),
                        'deptime': get_time_from_datetime_str(getattr(m, 'deptime', 'N/A')),
                        'arrtime': get_time_from_datetime_str(getattr(m, 'arrtime', 'N/A')),
                        'journeydurationinmin': getattr(m, 'journeydurationinmin', 'N/A'),
                        'travelsname': getattr(m, 'travelsname', 'N/A'),
                        'bustype': getattr(m, 'bustype', 'N/A'),
                        'isseater': getattr(m, 'isseater', 'N/A'),
                        'issleeper': getattr(m, 'issleeper', 'N/A'),
                        'servicename': getattr(m, 'servicename', 'N/A'),
                        'serviceid': getattr(m, 'serviceid', 'N/A'),
                        'slid': getattr(m, 'slid', 'N/A'),
                        'cosine_similarity': round(getattr(m, 'cosine_similarity', 0), 4)
                    }
                    serializable_matches.append(match_dict)

                # Create detailed message with all route information
                message_parts = []
                if extracted_source or extracted_destination:
                    message_parts.append(f"Extracted locations - Source: {extracted_source or 'Not specified'}, Destination: {extracted_destination or 'Not specified'}")
                
                message_parts.append(f"\n**TOP 5 MATCHING BUS ROUTES:**\n")
                
                for match in serializable_matches:
                    route_info = f"""
**Route #{match['rank']} - Route ID: {match['routeid']}**
• From: {match['sourcename']} ({match['sourceid']}) - {match['sourcestate']}
• To: {match['destinationname']} ({match['destinationid']}) - {match['destinationstate']}
• Departure: {match['deptime']} | Arrival: {match['arrtime']}
• Journey Duration: {match['journeydurationinmin']} minutes
• Travel Company: {match['travelsname']}
• Bus Type: {match['bustype']}
• Seating: {'Seater Available' if match['isseater'] else 'No Seater'} | {'Sleeper Available' if match['issleeper'] else 'No Sleeper'}
• Service: {match['servicename']} (Service ID: {match['serviceid']})
• SLID: {match['slid']}
• Similarity Score: {match['cosine_similarity']}
{'='*60}"""
                    message_parts.append(route_info)
                
                message_parts.append(f"\n**IMPORTANT:** Route IDs for further processing: {', '.join([str(match['routeid']) for match in serializable_matches])}")

                return RouteSearchResponse(
                    status="success",
                    message="\n".join(message_parts),
                    source_extracted=extracted_source,
                    destination_extracted=extracted_destination,
                    matches=serializable_matches
                )
            else:
                return RouteSearchResponse(
                    status="not_found",
                    message="No matching routes found for the provided search criteria.",
                    source_extracted=extracted_source,
                    destination_extracted=extracted_destination
                )
                
        finally:
            if session:
                session.cluster.shutdown()
                
    except Exception as e:
        logger.error(f"Error in Bus_route_search_tool: {e}", exc_info=True)
        return RouteSearchResponse(
            status="error",
            message=f"An error occurred while searching for routes: {str(e)}. Please try again later."
        )

# Create the LLM agent
Route_agent = LlmAgent(
    name="Route_Agent",
    model=LiteLlm(model="ollama_chat/llama3.2:latest"),
    tools=[Bus_route_search_tool],
    instruction="""
    You are a helpful bus route search assistant. Your primary goal is to search for bus routes based on user input and present the TOP 5 results with complete information for further processing.

    **Strict Steps:**
    1. **Extract User Intent:** Understand what the user is looking for in terms of bus routes (source, destination, preferences).
    2. **Call Tool:** Use the `Bus_route_search_tool` function with the user's text input.
    3. **Process Tool Result and Respond:**
        * If the tool returns `status: "success"`, present the detailed route information exactly as provided by the tool. The tool will format the top 5 routes with all necessary details including the critical Route IDs.
        * If the tool returns `status: "not_found"`, inform the user that no matching routes were found and suggest they try with different search criteria or locations.
        * If the tool returns `status: "error"`, acknowledge the error gracefully and suggest they try again later or rephrase their search query.
    4. **Handle Unclear Input:** If the user's input is too vague, ask for clarification about their travel preferences (source, destination, timing, etc.).

    **Critical Requirements:**
    - ALWAYS display exactly 5 routes (or fewer if less than 5 are found)
    - ALWAYS include the Route ID prominently for each route as it's critical for further processing
    - Present ALL available information for each route including: Route ID, source/destination details, timing, travel company, bus type, amenities, service details, and similarity scores
    - Include the summary of Route IDs at the end for easy reference
    - ***Only give output after the tool has successfully returned results, dont give results on your own , strictly return only results that were returned by the tool.***

    **Important:** The Route IDs are critical for downstream processing, so they must be clearly visible and easily extractable from your response. Always emphasize the Route IDs in your final response.
    """
)

# Set the Route_agent as the root agent for standalone execution
root_agent = Route_agent