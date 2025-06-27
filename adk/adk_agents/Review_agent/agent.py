import logging
import re
import json
import clickhouse_connect
from google.adk.agents import LlmAgent
from google.adk.models import Gemini # This import is not used in the provided code, but kept as is.

from google.adk.models.lite_llm import LiteLlm
from pydantic import BaseModel

# Define the response model for the tool
class ReviewSearchResponse(BaseModel):
    status: str
    message: str
    

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# ClickHouse configuration
# IMPORTANT: Ensure this IP address is accessible from where the agent is running.


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
        # Validate route_id is numeric
        if not route_id.isdigit():
            logger.warning(f"Invalid route ID received: '{route_id}'. Must be numeric.")
            return ReviewSearchResponse(
                status="error",
                message=f"Invalid route ID '{route_id}'. Route ID must be numeric."
            )
        
        # Establish connection to ClickHouse
        client = clickhouse_connect.get_client(
            host='10.5.40.193',
            port=8123,
            username='ugc_readonly',
            password='ugc@readonly!',
            secure=False
        )
        
        # SQL query to fetch one review for the given route ID
        # The query uses a subquery to find review IDs associated with the route ID.
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
        # Execute the query with parameters for safety and clarity
        result = client.query(query, parameters={"route_id": int(route_id)}).result_rows
        
        
            # If a review is found, extract it and return success
            
        logger.info(f"Review found for Route {route_id}. Review text snippet: '{result[:50]}...'")
        return ReviewSearchResponse(
            status="success",
            message=f"Review for Route {route_id}: {result}"
        )
        
            
    except Exception as e:
        # Catch any exceptions during the process and return an error status
        logger.error(f"Error fetching review from ClickHouse for Route {route_id}: {e}", exc_info=True)
        return ReviewSearchResponse(
            status="error",
            message=f"An error occurred while fetching review for Route {route_id}: {str(e)}. Please try again later."
        )

# Create the LLM agent
Review_agent = LlmAgent(
    name="Review_Agent",
    model=LiteLlm(model="ollama_chat/llama3.2:latest"), # Ensure this model is accessible and configured correctly
    tools=[Bus_review_tool], # Register the tool with the agent
    instruction="""
    You are a helpful bus review assistant. Your primary goal is to fetch and present a bus review for a given route ID, and then conclude your response for the current turn.

    **Strict Steps:**
    1.  **Extract Route ID:** Carefully identify and extract the numerical bus route ID from the user's message.
    2.  **Call Tool:** Use the `Bus_review_tool` function with the extracted route ID.
    3.  **Process Tool Result and Respond:**
        * If the tool returns a `status: "success"`, present the review text clearly summarised and politely. After presenting the review, **your task for this turn is complete, and you should provide a final, helpful response to the user.**
        * If the tool returns a `status: "not_found"`, inform the user that no reviews were found for that route ID, and politely offer to check another ID. **Then, your task for this turn is complete.**
        * If the tool returns an `status: "error"`, acknowledge the error gracefully and suggest they try again later or with a different route ID. **Then, your task for this turn is complete.**
    4.  **Handle Missing Route ID:** If the user does not provide a route ID in their initial message, politely ask them to specify one. Do NOT try to call the tool without a route ID.

    **Important:** Once you have provided a response based on the `Bus_review_tool`'s output (whether success, not found, or error), consider the current turn's objective achieved. **Do not re-prompt for the same route ID or re-call the tool for the same request.** Focus on providing a single, conclusive answer for each user query.
    """
)

# Set the Review_agent as the root agent for standalone execution
root_agent = Review_agent 
