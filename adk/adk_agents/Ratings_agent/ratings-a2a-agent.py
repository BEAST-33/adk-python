from fastapi import FastAPI, Request
from datetime import datetime
import logging
import json

# Make sure get_client is imported/available
# from your_clickhouse_module import get_client

def get_bus_rating(route_id):
    try:
        client = get_client(
            host='10.5.40.193',
            port=8123,
            username='ugc_readonly',
            password='ugc@readonly!',
            secure=False
        )
        query = "SELECT Value FROM UGC.UserRatings WHERE RouteID = %s"
        result = client.query(query, (route_id,))
        logging.debug("Results of the ratings: =>> %s", result)
        rows = result.result_rows
        if rows:
            logging.debug("Individual ratings: %s", rows)
            ratings = [row[0] for row in rows]
            average_rating = round(sum(ratings) / len(ratings), 1)
            logging.info(f"Average bus score for RouteID {route_id}: {average_rating}")
            return average_rating
        else:
            logging.info("No data found for RouteID %s", route_id)
            return None

    except Exception as e:
        logging.error(f"Error fetching bus rating from ClickHouse: {e}")
        return None

logging.basicConfig(level=logging.DEBUG)
app = FastAPI()

def make_a2a_response(request_id, task_id, state, message_text, rating=None):
    now = datetime.utcnow().isoformat()
    result = {
        "id": task_id,
        "status": {"state": state, "timestamp": now},
        "history": []
    }
    if state == "completed" and rating is not None:
        result["artifacts"] = [
            {"parts": [
                {"type": "text", "text": f"Average bus rating: {rating}"},
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
    import re
    # Now use regex on user_text, NOT parts!
    match = re.search(r'\d+', user_text)
    if match:
        route_id = match.group()
        rating = get_bus_rating(route_id)
        # ... continue as before ...
    else:
        error_message = "No valid route ID found in the input."
        logging.error(error_message)
        return make_a2a_response(request_id, task_id, "failed", error_message)

    if rating is not None:
        msg = f"Bus rating fetched for RouteID {route_id}."
        return make_a2a_response(request_id, task_id, "completed", msg, rating=rating)
    else:
        error_message = f"No rating found for RouteID {route_id}."
        logging.error(error_message)
        return make_a2a_response(request_id, task_id, "failed", error_message)

@app.post("/tasks/send")
async def tasks_send(request: Request):
    return await root(request)

@app.get("/.well-known/agent.json")
async def agent_manifest(request: Request):
    # Log the request details
    logging.info(f"Request received: {request.method} {request.url}")
    logging.info(f"Request headers: {dict(request.headers)}")

    response_content = {
        "name": "Bus Rating Agent",
        "description": (
            "Agent that accepts a bus route ID and returns the average user rating "
            "for that route, as stored in ClickHouse. Useful for retrieving feedback "
            "and quality metrics for bus services."
        ),
        "url": "http://10.5.10.190:8020/",
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
        "defaultInputModes": ["text", "text/plain", "application/json"],
        "defaultOutputModes": ["text", "text/plain", "application/json"],
        "skills": [
            {
                "id": "get_bus_rating",
                "name": "Get Bus Rating",
                "description": (
                    "Fetches the average user rating for a given bus route ID from ClickHouse. "
                    "Returns the rating as a floating-point number, or an error if not found."
                ),
                "tags": ["bus", "rating", "clickhouse", "feedback", "quality"],
                "examples": [
                    "Get the rating for route ID '123'.",
                    "What is the average rating for bus route 456?"
                ],
                "inputModes": ["text", "text/plain", "application/json"],
                "outputModes": ["text", "text/plain", "application/json"],
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
                                            "text": {"type": "string", "description": "The bus route ID to fetch the rating for."}
                                        },
                                        "required": ["type", "text"]
                                    },
                                    "description": "The list of message parts, with at least one containing the route ID."
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
                        "rating": {
                            "type": "number",
                            "description": "The average user rating for the requested route."
                        },
                        "route_id": {
                            "type": "string",
                            "description": "The bus route ID that was queried."
                        }
                    },
                    "required": ["rating", "route_id"]
                }
            }
        ]
    }

    return response_content
def get_client(host, port, username, password, secure):
    from clickhouse_connect import get_client as ch_get_client
    return ch_get_client(
        host=host,
        port=port,
        username=username,
        password=password,
        secure=secure
    )
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("ratings-a2a-agent:app", host="10.5.10.190", port=8020, reload=True)
