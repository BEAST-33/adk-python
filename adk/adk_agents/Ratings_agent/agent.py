import logging
import re
from pydantic import BaseModel
import clickhouse_connect
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- ClickHouse Client Helper (similar to your original, but ensure import for clickhouse_connect) ---
def get_client(host, port, username, password, secure):
    """Establishes and returns a ClickHouse client connection."""
    return clickhouse_connect.get_client(
        host=host,
        port=port,
        username=username,
        password=password,
        secure=secure
    )

# --- Define the Response Model for the Tool ---
class BusRatingResponse(BaseModel):
    """
    Represents the response structure for the get_bus_rating tool.
    """
    status: str  # e.g., "success", "not_found", "error"
    message: str # A descriptive message
    route_id: str
    average_rating: float | None = None # Use None for cases where rating isn't found

# --- Tool Function for ADK Agent ---
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
        # Basic validation for route_id
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

# --- Create the LLM agent ---
Ratings_agent = LlmAgent(
    name="Bus_Rating_Agent",
    model=LiteLlm(model="ollama_chat/llama3.2:latest"), # Ensure this model is accessible and configured correctly
    tools=[get_bus_rating_tool], # Register the new tool
    instruction="""
    You are a helpful bus rating assistant. Your primary goal is to fetch and present the average bus rating for a given route ID, and then conclude your response for the current turn.

    **Strict Steps:**
    1.  **Extract Route ID:** Carefully identify and extract the numerical bus route ID from the user's message.
        If the user asks for a bus rating, your first action is to extract the route ID. Use regular expressions if necessary to find digits in the user's input.
        Example patterns: "bus route 123", "rating for 456", "route ID is 789".
    2.  **Call Tool:** Use the `get_bus_rating_tool` function with the extracted `route_id`.
    3.  **Process Tool Result and Respond:**
        * If the tool returns a `status: "success"`, present the `average_rating` clearly, rounded to one decimal place, and politely. After presenting the rating, **your task for this turn is complete, and you should provide a final, helpful response to the user.**
        * If the tool returns a `status: "not_found"`, inform the user that no ratings were found for that route ID, and politely offer to check another ID. **Then, your task for this turn is complete.**
        * If the tool returns an `status: "error"`, acknowledge the error gracefully and suggest they try again later or with a different route ID. **Then, your task for this turn is complete.**
    4.  **Handle Missing Route ID:** If the user does not provide a route ID in their initial message, politely ask them to specify one. Do NOT try to call the tool without a route ID.

    **Important:** Once you have provided a response based on the `get_bus_rating_tool`'s output (whether success, not found, or error), consider the current turn's objective achieved. **Do not re-prompt for the same route ID or re-call the tool for the same request.** Focus on providing a single, conclusive answer for each user query.
    """
)

# Set the Ratings_agent as the root agent for standalone execution
root_agent = Ratings_agent

