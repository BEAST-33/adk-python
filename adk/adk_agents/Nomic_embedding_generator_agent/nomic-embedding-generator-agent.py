from fastapi import FastAPI, Request
from datetime import datetime
import json
import urllib.parse
import http.client
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)  # Set logging level to DEBUG

app = FastAPI()

def make_post_request(url, payload, headers):
    try:
        logging.debug(f"Making POST request to: {url}")  # Log the URL
        logging.debug(f"Payload: {payload}")  # Log the payload
        #logging.debug(f"Headers: {headers}")  # Log the headers

        parsed_url = urllib.parse.urlparse(url)
        conn = http.client.HTTPConnection(parsed_url.hostname, parsed_url.port or 80)
        conn.request("POST", parsed_url.path, json.dumps(payload), headers)
        response = conn.getresponse()
        data = response.read().decode('utf-8')

        logging.debug(f"Response status: {response.status}")  # Log the response status
        logging.debug(f"Response data: {data}")  # Log the response data

        conn.close()
        return json.loads(data)
    except Exception as e:
        logging.error(f"Error making HTTP request to {url}: {e}")
        return None

def generate_nomic_embeddings(text, api_url=None, model_name=None):
    if api_url is None:
        api_url = "http://10.166.8.126:11434/api/embed"
    if model_name is None:
        model_name = "nomic-embed-text:latest"
    payload = {"model": model_name, "input": str(text)}
    headers = {'Content-Type': 'application/json'}
    #logging.debug(f"Generating embeddings for text: {text}") # Log input text
    result = make_post_request(api_url, payload, headers)
    if not result:
        return None
    embeddings = result.get("data", [])
    embeddings = result.get("embeddings", [])
    if isinstance(embeddings, list) and len(embeddings) > 0:
        return embeddings[0]
    return None

def make_a2a_response(request_id, task_id, state, message_text, embedding=None):
    now = datetime.utcnow().isoformat()
    result = {
        "id": task_id,
        "status": {"state": state, "timestamp": now},
        "history": []
    }
    if state == "completed" and embedding is not None:
        result["artifacts"] = [
            {"parts": [
                {"type": "text", "text": str(embedding)},
               # {"type": "json", "json": str(embedding)}
            ], "index": 0}
        ]
        #logging.debug(f"make_a2a_response with embedding: {result}") # Log the result
    else:
        result["artifacts"] = [
            {"parts": [
                {"type": "text", "text": message_text}
            ], "index": 0}
        ]
        #logging.debug(f"make_a2a_response without embedding: {result}") # Log the result
    return {"jsonrpc": "2.0", "id": request_id, "result": result}

@app.post("/")
async def root(request: Request):
    payload = await request.json()
    logging.debug(f"Received payload: {payload}")  # Log the entire payload
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
        error_message = "No text provided."
        logging.error(error_message)
        return make_a2a_response(request_id, task_id, "failed", error_message)

    embedding = generate_nomic_embeddings(user_text)
    if embedding:
        msg = f"Embedding generated. Length: {len(embedding)}"
        #logging.info(msg)
        return make_a2a_response(request_id, task_id, "completed", msg, embedding=embedding)
    else:
        error_message = "Could not generate embedding."
        logging.error(error_message)
        return make_a2a_response(request_id, task_id, "failed", error_message)

@app.post("/tasks/send")
async def tasks_send(request: Request):
    return await root(request)

@app.get("/.well-known/agent.json")
async def agent_manifest():
    return {
        "name": "Nomic Embedding Agent",
        "description": "Agent that generates and returns the full Nomic embedding for given text.",
        "url": "http://10.5.10.190:8010/",
        "version": "1.0.0",
        "skills": [
            {
                "id": "generate_nomic_embedding",
                "name": "Generate Nomic Embedding",
                "description": "Returns the full Nomic embedding for a given text.",
                "tags": ["embedding", "nomic"],
                "examples": ["Bangalore to Chennai"]
            }
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("nomic-embedding-generator-agent:app", host="10.5.10.190", port=8010, reload=True)
