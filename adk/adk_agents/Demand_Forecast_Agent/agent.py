import logging
import re
import json
import mysql.connector
from datetime import datetime
from typing import Optional, List, Dict
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from pydantic import BaseModel

# Define the response model for the tool
class DemandForecastResponse(BaseModel):
    status: str
    message: str
    data: Optional[List[Dict]] = None

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# MySQL RDS connection details
RDS_ENDPOINT = "dp-replica-rds.redbus.com"
RDS_USER = "dpreaduser"
RDS_PASSWORD = "DpR3@du782^da23"
RDS_PORT = 3306
RDS_DB_NAME = "dp"

def extract_filters_from_text(text: str) -> Optional[Dict]:
    """
    Extracts source_name, destination_name, start_date, and end_date
    from a plain English sentence.

    Examples:
    - "Demand Forecast for Bangalore to Chennai from 2025-05-21 to 2025-05-25"
    - "Demand Forecast for Bir (Himachal Pradesh) to Delhi from 2025-06-03 to 2025-07-25"

    Returns a dict with parsed values including 'source_name', 'destination_name',
    'start_date', and 'end_date'.
    Returns None if parsing fails.
    """
    # Pattern: "for <source> to <destination> from <start_date> to <end_date>"
    pattern = (
        r"for\s+([A-Za-z\s()]+)\s+" # Source (Group 1) - allows parentheses
        r"to\s+([A-Za-z\s()]+)\s+"  # Destination (Group 2) - allows parentheses
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
        return filters
    return None

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
        logger.error(f"MySQL query error: {e}")
        raise
    finally:
        if conn:
            conn.close()

def demand_forecast_tool(user_input: str) -> DemandForecastResponse:
    """
    Fetch demand forecast data filtered by source, destination, and date range from MySQL RDS.
    
    Args:
        user_input: Natural language input or JSON string containing source_name, destination_name, 
                   start_date, and end_date parameters.
        
    Returns:
        DemandForecastResponse: An object containing the status of the operation 
                               ('success', 'not_found', 'error'), a message, and optional data.
    """
    logger.info(f"demand_forecast_tool called with user_input: {user_input}")
    
    try:
        # Parse input - try JSON first, then natural language
        parsed_input = None
        try:
            parsed_input = json.loads(user_input)
        except Exception:
            # If not JSON, try to extract from natural language
            parsed_input = extract_filters_from_text(user_input)
            if not parsed_input:
                return DemandForecastResponse(
                    status="error",
                    message="Invalid input: could not parse filter parameters from message text. Please provide input in format: 'Demand Forecast for [source] to [destination] from YYYY-MM-DD to YYYY-MM-DD'"
                )

        source_name = parsed_input.get("source_name")
        destination_name = parsed_input.get("destination_name")
        start_date = parsed_input.get("start_date")
        end_date = parsed_input.get("end_date")

        # Validate required parameters
        missing_params = []
        for param_name, param_value in [("source_name", source_name), ("destination_name", destination_name), ("start_date", start_date), ("end_date", end_date)]:
            if not param_value:
                missing_params.append(param_name)
        
        if missing_params:
            return DemandForecastResponse(
                status="error",
                message=f"Missing required parameters: {', '.join(missing_params)}"
            )

        # Validate date format YYYY-MM-DD
        try:
            datetime.strptime(start_date, "%Y-%m-%d")
            datetime.strptime(end_date, "%Y-%m-%d")
        except ValueError:
            return DemandForecastResponse(
                status="error",
                message="Dates must be in YYYY-MM-DD format."
            )

        # Query the database
        rows = query_demand_forecast_filtered(source_name, destination_name, start_date, end_date)
        
        if rows:
            logger.info(f"Found {len(rows)} records for {source_name} to {destination_name}")
            return DemandForecastResponse(
                status="success",
                message=f"Successfully fetched {len(rows)} demand forecast records for {source_name} to {destination_name} between {start_date} and {end_date}.",
                data=rows
            )
        else:
            return DemandForecastResponse(
                status="not_found",
                message=f"No demand forecast data found for {source_name} to {destination_name} between {start_date} and {end_date}."
            )
            
    except Exception as e:
        logger.error(f"Error in demand_forecast_tool: {e}", exc_info=True)
        return DemandForecastResponse(
            status="error",
            message=f"An error occurred while fetching demand forecast data: {str(e)}. Please try again later."
        )

# Create the LLM agent
Demand_Forecast_Agent = LlmAgent(
    name="DBD_Agent",
    model=LiteLlm(model="ollama_chat/llama3.2:latest"), # Ensure this model is accessible and configured correctly
    tools=[demand_forecast_tool], # Register the tool with the agent
    instruction="""
    You are a helpful demand forecast assistant for bus routes. Your primary goal is to fetch and present demand forecast data for specified routes and date ranges.

    **Strict Steps:**
    
    1.  **Call Tool:** Use the `demand_forecast_tool` function with the user's input (either as natural language or JSON) which ever format the user as entered you can input it directly into the .
    2.  **Process Tool Result and Respond:**
        * If the tool returns `status: "success"`, present the demand forecast data in a clear, organized format. Summarize key insights from the data such as peak demand periods, average predictions, etc. **After presenting the data, your task for this turn is complete.**
        * If the tool returns `status: "not_found"`, inform the user that no demand forecast data was found for the specified criteria. Suggest checking the spelling of city names or trying different date ranges. **Then, your task for this turn is complete.**
        * If the tool returns `status: "error"`, acknowledge the error gracefully and provide the specific error message. Suggest they try again with correct format or contact support if the issue persists. **Then, your task for this turn is complete.**
    3.  **Handle Incomplete Requests:** If the user doesn't provide all required information (source, destination, start date, end date), politely ask them to specify the missing details. Provide an example of the expected format.

    **Input Format Examples:**
    - "Demand Forecast for Mumbai to Pune from 2025-06-01 to 2025-06-15"
    - "Show me demand forecast for Bangalore to Chennai from 2025-07-01 to 2025-07-31"
    - JSON: {"source_name": "Delhi", "destination_name": "Gurgaon", "start_date": "2025-06-01", "end_date": "2025-06-30"}

    **Important:** Once you have provided a response based on the `demand_forecast_tool`'s output (whether success, not found, or error), consider the current turn's objective achieved. Focus on providing a single, conclusive answer for each user query.
    """
)

# Set the DBD_agent as the root agent for standalone execution
root_agent = Demand_Forecast_Agent