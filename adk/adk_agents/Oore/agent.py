import requests
import re
import logging
import json
from datetime import datetime
from pydantic import BaseModel
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm

# Configure robust logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("redbus-adk-agent")

REDBUS_API_URL = "http://oore-india.redbus.com:6666/srp/"

# --- Define the Response Model for the Tool ---
class RedBusDealsResponse(BaseModel):
    """
    Represents the response structure for the get_redbus_deals_tool.
    """
    status: str  # e.g., "success", "no_data", "error", "invalid_input"
    message: str # A descriptive message
    bus_summary: str | None = None # Summarized bus information

# --- Tool Function for ADK Agent ---
def get_redbus_deals_tool(source_id: int, dest_id: int, doj: str, chcode: str = "WEB_DIRECT") -> RedBusDealsResponse:
    """
    Fetches RedBus schedules and deals for a given source, destination, and date.
    Summarizes the top few bus options.

    Args:
        source_id: The numerical ID of the source city/location.
        dest_id: The numerical ID of the destination city/location.
        doj: The date of journey in 'YYYY-MM-DD' format.
        chcode: The channel code, defaults to "WEB_DIRECT".

    Returns:
        RedBusDealsResponse: An object containing the status, a message, and
                             a summary of available buses or an error/no data message.
    """
    logger.info(f"get_redbus_deals_tool called with s={source_id}, d={dest_id}, doj={doj}, chcode={chcode}")

    # Basic input validation for tool
    if not isinstance(source_id, int) or not isinstance(dest_id, int) or not doj:
        return RedBusDealsResponse(
            status="invalid_input",
            message="Invalid source ID, destination ID, or date provided. Please ensure they are correct.",
            bus_summary=None
        )

    # Validate DOJ format (simple check, full validation can be more robust if needed)
    try:
        datetime.strptime(doj, "%Y-%m-%d")
    except ValueError:
        return RedBusDealsResponse(
            status="invalid_input",
            message="Invalid date format. Please provide the date in 'YYYY-MM-DD' format (e.g., 2025-06-26).",
            bus_summary=None
        )

    params = {
        "slc": 1,
        "d": dest_id,
        "s": source_id,
        "chcode": chcode,
        "doj": doj,
        "c": "IND"
    }

    try:
        logger.info(f"Calling RedBus API with params: {params}")
        resp = requests.get(REDBUS_API_URL, params=params, timeout=10)
        resp.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)

        data = resp.json()

        if not data or not data.get("Bus"):
            logger.info(f"No bus data found for s={source_id}, d={dest_id}, doj={doj}")
            return RedBusDealsResponse(
                status="no_data",
                message=f"No buses found for source {source_id} to destination {dest_id} on {doj}.",
                bus_summary="No buses found for your request."
            )

        # Summarize buses (keeping your original logic)
        lines = []
        bus_count = 0
        for bus_id, bus in data.get("Bus", {}).items():
            if bus_count >= 5: # Limit to top 5 options
                break
            op = f"₹{min(bus['OP'])/100:.2f}" if bus.get("OP") and isinstance(bus.get("OP"), list) and len(bus.get("OP")) > 0 else "N/A"
            dp = f"₹{min(bus['DP'])/100:.2f}" if bus.get("DP") and isinstance(bus.get("DP"), list) and len(bus.get("DP")) > 0 else "N/A"
            seat = bus.get("SeatCount", "N/A")
            lines.append(f"Bus {bus_id}: Price {op} (Discounted {dp}), Seats: {seat}")
            bus_count += 1
        
        summary = "\n".join(lines) if lines else "No buses found that match summary criteria."
        
        logger.info(f"RedBus API call successful and summarized data for s={source_id}, d={dest_id}, doj={doj}.")
        return RedBusDealsResponse(
            status="success",
            message="Successfully fetched RedBus deals.",
            bus_summary=summary
        )

    except requests.exceptions.Timeout:
        logger.error(f"RedBus API request timed out for s={source_id}, d={dest_id}, doj={doj}")
        return RedBusDealsResponse(
            status="error",
            message="The request to RedBus API timed out. Please try again later.",
            bus_summary=None
        )
    except requests.exceptions.RequestException as e:
        logger.error(f"RedBus API network error for s={source_id}, d={dest_id}, doj={doj}: {e}", exc_info=True)
        return RedBusDealsResponse(
            status="error",
            message=f"A network error occurred while fetching RedBus deals: {str(e)}",
            bus_summary=None
        )
    except json.JSONDecodeError:
        logger.error(f"RedBus API returned invalid JSON for s={source_id}, d={dest_id}, doj={doj}")
        return RedBusDealsResponse(
            status="error",
            message="Received an unreadable response from the RedBus service. Please try again.",
            bus_summary=None
        )
    except Exception as e:
        logger.error(f"An unexpected error occurred in get_redbus_deals_tool for s={source_id}, d={dest_id}, doj={doj}: {e}", exc_info=True)
        return RedBusDealsResponse(
            status="error",
            message=f"An unexpected error occurred: {str(e)}. Please try again later.",
            bus_summary=None
        )

# --- Create the LLM agent ---
RedBusDealsAgent = LlmAgent(
    name="RedBus_Deals_Agent",
    model=LiteLlm(model="ollama_chat/llama3.2:latest"), # Ensure this model is accessible and configured correctly
    tools=[get_redbus_deals_tool], # Register the new tool
    instruction="""
    You are a helpful RedBus deals assistant. Your primary goal is to assist users in finding bus schedules and deals
    by extracting the necessary information (source ID, destination ID, and date of journey) from their queries,
    calling the appropriate tool, and presenting the results.

    **Strict Steps:**
    1.  **Extract Parameters:** Carefully listen for and extract three key pieces of information from the user's message:
        * `source_id`: A numerical ID for the origin city/location.
        * `dest_id`: A numerical ID for the destination city/location.
        * `doj` (date of journey): The date, which you must convert to 'YYYY-MM-DD' format (e.g., '2025-06-26').
        * `chcode` (channel code): An optional string, default to "WEB_DIRECT" if not specified.
        
        Look for patterns like: "source: 123, destination: 124, date: 27th May 2025" or similar variations.
        Be robust in extracting these values. For the date, be sure to parse various formats (e.g., "27th May 2025", "May 27, 2025", "2025-05-27") and convert them to 'YYYY-MM-DD'.

    2.  **Call Tool:** Once you have *all three required parameters* (`source_id`, `dest_id`, `doj`), call the `get_redbus_deals_tool` with these arguments.

    3.  **Process Tool Result and Respond:**
        * If the tool returns a `status: "success"`, present the `bus_summary` clearly and politely to the user. After presenting the summary, **your task for this turn is complete, and you should provide a final, helpful response to the user.**
        * If the tool returns a `status: "no_data"`, inform the user that no buses were found for their specified criteria and politely offer to check other dates or routes. **Then, your task for this turn is complete.**
        * If the tool returns a `status: "invalid_input"`, inform the user about the invalid input (e.g., missing ID, wrong date format) and ask them to correct it. **Then, your task for this turn is complete.**
        * If the tool returns a `status: "error"`, acknowledge the error gracefully and suggest they try again later or with different parameters. **Then, your task for this turn is complete.**

    4.  **Handle Missing Parameters:** If the user does not provide *all three required parameters* in their initial message (source ID, destination ID, date), politely ask them to specify the missing information. Do NOT try to call the tool without all required parameters.

    **Important:** Once you have provided a response based on the `get_redbus_deals_tool`'s output, consider the current turn's objective achieved. **Do not re-prompt for the same query or re-call the tool for the same request.** Focus on providing a single, conclusive answer for each user query.
    """
)

# Set the RedBusDealsAgent as the root agent for standalone execution
root_agent = RedBusDealsAgent

# To run this agent, save this file (e.g., as redbus_adk_agent.py) and execute:
# python -m google.adk.cmd.run redbus_adk_agent.py
# Make sure your Ollama server is running and the RedBus API is accessible.