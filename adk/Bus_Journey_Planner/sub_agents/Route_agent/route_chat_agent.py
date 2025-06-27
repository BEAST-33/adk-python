from cassandra.cluster import Cluster
from cassandra.policies import RetryPolicy, ConstantReconnectionPolicy
import configparser
import os
import json
import urllib.parse
import http.client
import logging
import requests
import tiktoken
from datetime import datetime
from fastapi import FastAPI, Request
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
log_level = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(
    level=getattr(logging, log_level),
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI()

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
    """Generate embeddings using either local Nomic API or GIR API based on configuration"""
    # First try the default Nomic API
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
            logger.warning(f"Nomic API request failed with status ")
    except Exception as e:
        logger.warning(f"Error in Nomic embedding generation: {e}")
        return None
    
    

def perform_ann_query(session_param, table_name, query_embedding, limit=5):
    """
    Performs an Approximate Nearest Neighbors (ANN) query on the specified Cassandra table.

    Args:
        session_param: The Cassandra session to use.
        table_name (str): The name of the Cassandra table to query.
        query_embedding (list): The embedding vector to use for the ANN query.
        limit (int, optional): The maximum number of results to return. Defaults to 5.

    Returns:
        list: A list of rows representing the query results, or an empty list on error.    
    """
    if not session_param:
        logger.error("No valid Cassandra session provided to perform_ann_query.")
        return []

    if not query_embedding:
        logger.error("No query embedding provided for ANN query.")
        return []

    try:
        logger.info(f"Performing ANN query on table {table_name}...")

        # Find the embedding column
        query = f"""SELECT column_name FROM system_schema.columns 
                   WHERE keyspace_name = '{session_param.keyspace}' 
                   AND table_name = '{table_name}'"""
        rows = session_param.execute(query)
        embedding_col = None
        for row in rows:
            if row.column_name.endswith('_embedding'):
                embedding_col = row.column_name
                break

        if not embedding_col:
            logger.error(f"No embedding column found in table {table_name}")
            return []

        # Construct and execute the ANN query
        query = f"""SELECT * FROM {session_param.keyspace}.{table_name} 
                   ORDER BY {embedding_col} ANN OF {query_embedding} LIMIT {limit}"""
        results = session_param.execute(query)

        # Convert the results to a list of rows
        list_of_rows = list(results)
        logger.info(f"ANN query returned {len(list_of_rows)} results.")
        return list_of_rows

    except Exception as e:
        logger.error(f"Error performing ANN query: {e}")
        return []

def _load_system_prompt(prompt_file="route-system-prompt.txt"):
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        prompt_path = os.path.join(current_dir, prompt_file)
        if not os.path.exists(prompt_path):
            logger.error(f"System prompt file not found: {prompt_path}")
            return "You are a helpful assistant that provides information about routes."
        with open(prompt_path, 'r') as f:
            return f.read()
    except Exception as e:
        logger.error(f"Error loading system prompt: {e}")
        return "You are a helpful assistant that provides information about routes."
        
def generate_chat_response(user_message):
    api_url = os.getenv("CHAT_API", "http://10.120.17.147:11434/api/chat")
    model_name = os.getenv("CHAT_MODEL", "qwen2.5-coder:14b")
    system_prompt = _load_system_prompt()
    
    token_usage = {"system_tokens": 0, "user_tokens": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    try:
        system_tokens = count_tokens(system_prompt)
        user_tokens = count_tokens(user_message)
        total_input_tokens = system_tokens + user_tokens
        token_usage["system_tokens"] = system_tokens
        token_usage["user_tokens"] = user_tokens
        token_usage["input_tokens"] = total_input_tokens
        logger.info(f"Input tokens - System: {system_tokens}, User: {user_tokens}, Total: {total_input_tokens}")
        
        logger.info("Generating embedding for user query...")
        query_embedding = generate_nomic_embeddings(user_message)
        logger.info(f"Query embedding generated: {query_embedding is not None}")
        
        context_info = ""
        try:
            if query_embedding is not None:
                logger.info("Performing ANN query with embedding...")
                # Get the Cassandra session
                cassandra_session, _ = get_cassandra_session()
                if cassandra_session is None:
                    logger.error("Cassandra session is None. Cannot perform ANN query.")
                    return "Error: Cassandra session is not available.", token_usage
                
                table_name = os.getenv("CASSANDRA_TABLE", "dpe_route_embedding")
                ann_results = perform_ann_query(cassandra_session, table_name, query_embedding, limit=5)
                results_list = ann_results
                logger.info(f"ANN query returned {len(results_list)} results")
                
                if results_list:
                    context_info = "Relevant routes based on your query:\n\n"
                    for i, row in enumerate(results_list):
                        context_info += f"Route {i+1}:\n"
                        for col_name in row._fields:
                            if col_name != 'route_embedding' and hasattr(row, col_name):
                                value = getattr(row, col_name)
                                context_info += f"{col_name}: {value}\n"
                        context_info += "\n"
        except Exception as e:
            logger.error(f"Error performing ANN query: {e}")
            
        enhanced_user_message = user_message
        if context_info:
            enhanced_user_message = f"{user_message}\n\n{context_info}"
            enhanced_tokens = count_tokens(enhanced_user_message)
            token_usage["user_tokens"] = enhanced_tokens
            token_usage["input_tokens"] = token_usage["system_tokens"] + enhanced_tokens
            
        payload = {
            "stream": False, 
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": enhanced_user_message}
            ], 
            "model": model_name
        }
        headers = {'Content-Type': 'application/json'}
        
        logger.info("Making API call to generate response...")
        response = requests.post(api_url, headers=headers, json=payload)
        response.raise_for_status()
        
        response_json = response.json()
        response_content = response_json["message"]["content"]
        
        output_tokens = count_tokens(response_content)
        token_usage["output_tokens"] = output_tokens
        token_usage["total_tokens"] = token_usage["input_tokens"] + output_tokens
        
        logger.info(f"Output tokens: {output_tokens}")
        logger.info(f"Total tokens: {token_usage['total_tokens']}")
        
        return response_content, token_usage
    except Exception as e:
        logger.error(f"Error in generate_chat_response: {e}")
        return f"Error generating response: {str(e)}", token_usage

def count_tokens(text, model="cl100k_base"):
    try:
        encoding = tiktoken.get_encoding(model)
        return len(encoding.encode(text))
    except Exception as e:
        logger.error(f"Error counting tokens: {e}")
        return 0

def make_a2a_response(request_id, task_id, state, message_text, results=None):
    now = datetime.utcnow().isoformat()
    result = {
        "id": task_id,
        "status": {"state": state, "timestamp": now},
        "history": []
    }
    
    if state == "completed" and results is not None:
        result["artifacts"] = [
            {"parts": [
                {"type": "json", "json": results}
            ], "index": 0}
        ]
    else:
        result["artifacts"] = [
            {"parts": [
                {"type": "text", "text": message_text}
            ], "index": 0}
        ]
    return {"jsonrpc": "2.0", "id": request_id, "result": result}

@app.post("/")
async def root(request: Request):
    payload = await request.json()
    logger.debug(f"Received payload: {payload}")
    
    params = payload.get("params", payload)
    task_id = params.get("id")
    request_id = payload.get("id") or params.get("id")
    message = params.get("message", {})
    parts = message.get("parts", [])
    
    if isinstance(parts, dict):
        user_text = parts.get("text", "")
    elif isinstance(parts, list) and parts:
        user_text = parts[0].get("text", "") if isinstance(parts[0], dict) else parts[0]
    else:
        user_text = ""
        
    if not user_text:
        logger.error("No user text provided in the request.")
        error_message = "No text provided."
        return make_a2a_response(request_id, task_id, "failed", error_message)

    response_content, token_usage = generate_chat_response(user_text)
    if response_content:
        logger.info(f"Generated response: {response_content[::]}...")
        return make_a2a_response(request_id, task_id, "completed", response_content)
    else:
        logger.error("Failed to generate response.")
        error_message = "Error generating response."
        return make_a2a_response(request_id, task_id, "failed", error_message)     
        
@app.post("/tasks/send")
async def tasks_send(request: Request):
    return await root(request)
        
@app.get("/.well-known/agent.json")
async def agent_manifest():
    return {
        "name": "RouteChatAgent",
        "description": "Agent that returns top routes based on user query.",
        "url": os.getenv("PUBLIC_URL", "http://localhost:8010/"),
        "version": "1.0.0",
        "skills": [
            {
                "id": "route_chat_agent",
                "name": "Route chat agent",
                "description": "Generates the top routes based on user query.",
                "tags": ["routes"],
                "examples": ["Bangalore to Chennai"]
            }
        ]
    }

# Healthcheck endpoint - useful for monitoring
@app.get("/health")
async def health():
    # Test Cassandra connection
    session, cluster = get_cassandra_session()
    if session is None:
        return {"status": "error", "message": "Cassandra connection failed"}
    
    return {"status": "healthy"}

if __name__ == '__main__':
    import uvicorn
    
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8020"))
    
    uvicorn.run(app, host=host, port=port)