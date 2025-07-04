# bus_review_mcp_server.py
import asyncio
import json
import os
from dotenv import load_dotenv
import logging
import clickhouse_connect
from pydantic import BaseModel

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

# Define the response model for the tool
class ReviewSearchResponse(BaseModel):
    status: str
    message: str

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
        
        # SQL query to fetch reviews for the given route ID
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
        
        if result:
            # Extract all reviews and combine them
            reviews = []
            for row in result:
                if row and row[0]:  # Check if row exists and has content
                    reviews.append(str(row[0]).strip())
            
            if reviews:
                # Combine all reviews with proper formatting
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
            # No reviews found for this route ID
            logger.info(f"No reviews found for Route {route_id}")
            return ReviewSearchResponse(
                status="not_found",
                message=f"No reviews found for Route {route_id}"
            )
            
    except Exception as e:
        # Catch any exceptions during the process and return an error status
        logger.error(f"Error fetching review from ClickHouse for Route {route_id}: {e}", exc_info=True)
        return ReviewSearchResponse(
            status="error",
            message=f"An error occurred while fetching review for Route {route_id}: {str(e)}. Please try again later."
        )

# --- Prepare the ADK Tool ---
# Instantiate the ADK tool you want to expose.
# This tool will be wrapped and called by the MCP server.
print("Initializing ADK Bus_review_tool...")
adk_tool_to_expose = FunctionTool(Bus_review_tool)
print(f"ADK tool '{adk_tool_to_expose.name}' initialized and ready to be exposed via MCP.")
# --- End ADK Tool Prep ---

# --- MCP Server Setup ---
print("Creating MCP Server instance...")
# Create a named MCP Server instance using the mcp.server library
app = Server("bus-review-mcp-server")

# Implement the MCP server's handler to list available tools
@app.list_tools()
async def list_mcp_tools() -> list[mcp_types.Tool]:
    """MCP handler to list tools this server exposes."""
    print("MCP Server: Received list_tools request.")
    # Convert the ADK tool's definition to the MCP Tool schema format
    mcp_tool_schema = adk_to_mcp_tool_type(adk_tool_to_expose)
    print(f"MCP Server: Advertising tool: {mcp_tool_schema.name}")
    return [mcp_tool_schema]

# Implement the MCP server's handler to execute a tool call
@app.call_tool()
async def call_mcp_tool(
    name: str, arguments: dict
) -> list[mcp_types.Content]: # MCP uses mcp_types.Content
    """MCP handler to execute a tool call requested by an MCP client."""
    print(f"MCP Server: Received call_tool request for '{name}' with args: {arguments}")

    # Check if the requested tool name matches our wrapped ADK tool
    if name == adk_tool_to_expose.name:
        try:
            # Execute the ADK tool's run_async method.
            # Note: tool_context is None here because this MCP server is
            # running the ADK tool outside of a full ADK Runner invocation.
            # If the ADK tool requires ToolContext features (like state or auth),
            # this direct invocation might need more sophisticated handling.
            adk_tool_response = await adk_tool_to_expose.run_async(
                args=arguments,
                tool_context=None,
            )
            print(f"MCP Server: ADK tool '{name}' executed. Response: {adk_tool_response}")

            # Handle Pydantic model response - convert to dict if needed
            if hasattr(adk_tool_response, 'model_dump'):
                # It's a Pydantic model, convert to dict
                response_dict = adk_tool_response.model_dump()
            elif hasattr(adk_tool_response, 'dict'):
                # Older Pydantic version
                response_dict = adk_tool_response.dict()
            else:
                # Assume it's already a dict or JSON-serializable
                response_dict = adk_tool_response

            # Format the ADK tool's response into an MCP-compliant format.
            response_text = json.dumps(response_dict, indent=2)
            # MCP expects a list of mcp_types.Content parts
            return [mcp_types.TextContent(type="text", text=response_text)]

        except Exception as e:
            print(f"MCP Server: Error executing ADK tool '{name}': {e}")
            # Return an error message in MCP format
            error_text = json.dumps({"error": f"Failed to execute tool '{name}': {str(e)}"})
            return [mcp_types.TextContent(type="text", text=error_text)]
    else:
        # Handle calls to unknown tools
        print(f"MCP Server: Tool '{name}' not found/exposed by this server.")
        error_text = json.dumps({"error": f"Tool '{name}' not implemented by this server."})
        return [mcp_types.TextContent(type="text", text=error_text)]

# --- MCP Server Runner ---
async def run_mcp_stdio_server():
    """Runs the MCP server, listening for connections over standard input/output."""
    # Use the stdio_server context manager from the mcp.server.stdio library
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        print("MCP Stdio Server: Starting handshake with client...")
        await app.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name=app.name, # Use the server name defined above
                server_version="0.1.0",
                capabilities=app.get_capabilities(
                    # Define server capabilities - consult MCP docs for options
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )
        print("MCP Stdio Server: Run loop finished or client disconnected.")

if __name__ == "__main__":
    print("Launching MCP Server to expose ADK Bus Review tool via stdio...")
    try:
        asyncio.run(run_mcp_stdio_server())
    except KeyboardInterrupt:
        print("\nMCP Server (stdio) stopped by user.")
    except Exception as e:
        print(f"MCP Server (stdio) encountered an error: {e}")
    finally:
        print("MCP Server (stdio) process exiting.")
# --- End MCP Server ---