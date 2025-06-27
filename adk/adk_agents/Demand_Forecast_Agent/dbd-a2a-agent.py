from fastapi import FastAPI, Request
from datetime import datetime
from typing import Optional, List, Dict
import mysql.connector
import json
import logging
import re

app = FastAPI()
logging.basicConfig(level=logging.DEBUG)

# MySQL RDS connection details
RDS_ENDPOINT = "dp-replica-rds.redbus.com"
RDS_USER = "dpreaduser"
RDS_PASSWORD = "DpR3@du782^da23"
RDS_PORT = 3306
RDS_DB_NAME = "dp"

def make_a2a_response(
        request_id: str,
        task_id: str,
        status: str,
        description: str,
        matches: Optional[List[Dict]] = None
) -> Dict:
    """
    Constructs a JSON-RPC 2.0 response in the A2A format.
    """
    now = datetime.utcnow().isoformat()
    if matches is not None and len(matches) > 0:
        text = description + "\n" + json.dumps(matches, indent=2, default=str)
    else:
        text = description

    response = {
        "jsonrpc": "2.0",
        "id": request_id,
        "result": {
            "id": task_id,
            "status": {
                "state": status,
                "timestamp": now
            },
            "history": [],
            "artifacts": [
                {
                    "parts": [
                        {
                            "type": "text",
                            "text": json.dumps(text)
                        }
                    ],
                    "index": 0
                }
            ]
        }
    }
    print("response ==> ", response)
    return response

def query_demand_forecast_filtered(
        source_name: str,
        destination_name: str,
        start_date: str,
        end_date: str
) -> List[Dict]:
    """
    Query demand forecast filtered by source, destination, and forecasting_date range.
    """
    conn = None
    try:
        conn = mysql.connector.connect(
            host=RDS_ENDPOINT,
            user=RDS_USER,
            password=RDS_PASSWORD,
            port=RDS_PORT,
            database=RDS_DB_NAME
        )
        cursor = conn.cursor(dictionary=True)
        query = """
            SELECT
              forecasting_date,
              doj,
              rb_source_id,
              rb_dest_id,
              early_sale_percentile,
              actual_early_sales,
              source_name,
              destination_name,
              dbd,
              pred_seatcount_pre_adjustment,
              adjustment_factor,
              seat_booked_till_date,
              last_updated_time,
              region_name,
              early_sale_percentile_region,
              pred_seatcount_pre_adjustment_region,
              pred_seatcount_post_adjustment_region
            FROM demand_forecast_new
            WHERE source_name = %s
              AND destination_name = %s
              AND forecasting_date BETWEEN %s AND %s
            ORDER BY forecasting_date ASC
            LIMIT 100
        """
        cursor.execute(query, (source_name, destination_name, start_date, end_date))
        rows = cursor.fetchall()
        return rows
    except Exception as e:
        logging.error(f"MySQL query error: {e}")
        raise
    finally:
        if conn:
            conn.close()

import re
from typing import Dict, Optional

def extract_filters_from_text(text: str) -> Optional[Dict]:
    """
    Extracts source_name, destination_name, start_date, and end_date
    from a plain English sentence. It no longer extracts state information
    from within parentheses.

    Examples:
    - "Demand Forecast for Bangalore to Chennai from 2025-05-21 to 2025-05-25"
    - "Demand Forecast for Bir (Himachal Pradesh) to Delhi from 2025-06-03 to 2025-07-25"
      (Note: '(Himachal Pradesh)' will now be treated as part of 'Bir' if present,
      but the regex below will aim to capture just the city name if possible,
      or the whole string including the bracketed part if it's simpler to implement
      without affecting the core functionality.)

    Returns a dict with parsed values including 'source_name', 'destination_name',
    'start_date', and 'end_date'.
    Returns None if parsing fails.
    """
    # Pattern: "for <source> to <destination> from <start_date> to <end_date>"
    # This pattern will capture anything between "for " and "to ", and "to " and "from "
    # as source_name and destination_name, respectively.
    # It does not attempt to parse out content within brackets.
    pattern = (
        r"for\s+([A-Za-z\s()]+)\s+" # Source (Group 1) - now explicitly allows parentheses
        r"to\s+([A-Za-z\s()]+)\s+"  # Destination (Group 2) - now explicitly allows parentheses
        r"from\s+(\d{4}-\d{2}-\d{2})\s+"                    # Start Date (Group 3)
        r"to\s+(\d{4}-\d{2}-\d{2})"                         # End Date (Group 4)
    )
    match = re.search(pattern, text, re.IGNORECASE)

    if match:
        filters = {
            "source_name": match.group(1).strip(),
            "destination_name": match.group(2).strip(),
            "start_date": match.group(3).strip(),
            "end_date": match.group(4).strip()
        }
        # No 'source_state' or 'destination_state' fields will be added.
        return filters
    return None

#def extract_filters_from_text(text: str):
#    """
#    Extracts source_name, destination_name, start_date, end_date from a plain English sentence.
#    Example: "Demand Forecast for Bangalore to Chennai from 2025-05-21 to 2025-05-25"
#    Returns a dict or None if parsing fails.
#    """
#    # Pattern: "for <source> to <destination> from <start_date> to <end_date>"
#    pattern = r"for\s+([A-Za-z\s]+)\s+to\s+([A-Za-z\s]+)\s+from\s+(\d{4}-\d{2}-\d{2})\s+to\s+(\d{4}-\d{2}-\d{2})"
#    match = re.search(pattern, text, re.IGNORECASE)
#    if match:
#        source_name = match.group(1).strip()
#        destination_name = match.group(2).strip()
#        start_date = match.group(3).strip()
#        end_date = match.group(4).strip()
#        return {
#            "source_name": source_name,
#            "destination_name": destination_name,
#            "start_date": start_date,
#            "end_date": end_date
#        }
#    return None

@app.post("/")
async def root(request: Request):
    payload = await request.json()
    logging.debug(f"Received payload: {payload}")
    params = payload.get("params", payload)
    task_id = params.get("id")
    request_id = payload.get("id") or params.get("id")

    message = params.get("message", {})
    parts = message.get("parts", [])

    if not parts or not isinstance(parts[0], dict):
        return make_a2a_response(request_id, task_id, "failed", "Invalid input format: missing message parts.")

    text = parts[0].get("text", "")
    user_input = None

    # Try to parse as JSON first
    try:
        user_input = json.loads(text)
    except Exception:
        # If not JSON, try to extract from natural language
        user_input = extract_filters_from_text(text)
        if not user_input:
            return make_a2a_response(request_id, task_id, "failed",
                                     "Invalid input: could not parse filter parameters from message text.")

    source_name = user_input.get("source_name")
    destination_name = user_input.get("destination_name")
    start_date = user_input.get("start_date")
    end_date = user_input.get("end_date")

    # Validate required parameters
    missing_params = []
    for param_name, param_value in [("source_name", source_name), ("destination_name", destination_name), ("start_date", start_date), ("end_date", end_date)]:
        if not param_value:
            missing_params.append(param_name)
    if missing_params:
        return make_a2a_response(request_id, task_id, "failed", f"Missing required parameters: {', '.join(missing_params)}")

    # Optional: Validate date format YYYY-MM-DD
    try:
        datetime.strptime(start_date, "%Y-%m-%d")
        datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError:
        return make_a2a_response(request_id, task_id, "failed", "Dates must be in YYYY-MM-DD format.")

    try:
        rows = query_demand_forecast_filtered(source_name, destination_name, start_date, end_date)
        if rows:
            description = f"Fetched {len(rows)} records for source '{source_name}', destination '{destination_name}' between {start_date} and {end_date}."
            return make_a2a_response(request_id, task_id, "completed", description, matches=rows)
        else:
            return make_a2a_response(request_id, task_id, "completed", "No data found for the specified filters.", matches=[])
    except Exception as e:
        return make_a2a_response(request_id, task_id, "failed", f"Error querying demand forecast: {str(e)}")

@app.post("/tasks/send")
async def tasks_send(request: Request):
    return await root(request)

@app.get("/.well-known/agent.json")
async def agent_manifest():
    return {
        "name": "Demand Forecast Agent",
        "description": "Agent that fetches demand forecast data filtered by source, destination, and date range from MySQL RDS.",
        "url": "http://10.5.10.190:8032/",
        "version": "1.0.0",
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
                "id": "get_demand_forecast_filtered",
                "name": "Get Demand Forecast Filtered",
                "description": "Fetches demand forecast records filtered by source, destination, and forecasting date range.",
                "tags": ["demand", "forecast", "mysql", "rds", "filter"],
                "examples": [
                    "Get demand forecast for source Mumbai, destination Pune, from 2025-05-01 to 2025-05-15.",
                    "Show me demand forecast records filtered by source and destination."
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
                                            "text": {
                                                "type": "string",
                                                "description": "JSON string with keys: source_name, destination_name, start_date, end_date"
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
                }
            }
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("dbd-a2a-agent:app", host="0.0.0.0", port=8032, reload=True)
