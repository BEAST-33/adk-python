# bus_review_rating_mcp_agent.py
import os
from google.adk.agents import LlmAgent
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, StdioServerParameters
from google.adk.models.lite_llm import LiteLlm

# IMPORTANT: Replace this with the ABSOLUTE path to your bus_review_mcp_server.py script
PATH_TO_YOUR_MCP_SERVER_SCRIPT = "/Users/karthik.mr/adk-python/adk/adk_agents/mcp_review_and_rating/bus_review_rating_mcp_server.py"  # <<< REPLACE WITH ACTUAL PATH

if PATH_TO_YOUR_MCP_SERVER_SCRIPT == "/path/to/your/bus_review_mcp_server.py":
    print("WARNING: PATH_TO_YOUR_MCP_SERVER_SCRIPT is not set. Please update it in bus_review_rating_mcp_agent.py.")
    # Optionally, raise an error if the path is critical
    # raise ValueError("Please set the correct path to your MCP server script")

# Create the LLM agent that uses the MCP server with both tools
Bus_Review_Rating_Agent = LlmAgent(
    name="Bus_Review_Rating_Agent_MCP",
    model=LiteLlm(model="ollama_chat/llama3.2:latest"),
    tools=[
        MCPToolset(
            connection_params=StdioServerParameters(
                command='python3',  # Command to run your MCP server script
                args=[PATH_TO_YOUR_MCP_SERVER_SCRIPT],  # Argument is the path to the script
            ),
            # tool_filter=['Bus_review_tool', 'get_bus_rating_tool']  # Optional: ensure only specific tools are loaded
        )
    ],
    instruction="""
    You are a comprehensive bus information assistant. You can help users with both bus reviews and bus ratings for specific route IDs.

    **Available Functions:**
    1. `Bus_review_tool` - Fetches user reviews for a bus route
    2. `get_bus_rating_tool` - Fetches average ratings for a bus route

    **Your Process:**
    1. **Understand the Request:** Determine what the user is asking for:
       - If they want reviews/feedback/comments → use `Bus_review_tool`
       - If they want ratings/scores/average rating → use `get_bus_rating_tool`
       - If they want both or it's unclear → ask for clarification

    2. **Extract Route ID:** Carefully identify and extract the numerical bus route ID from the user's message.

    3. **Call Appropriate Tool:** Use the relevant tool function with the extracted route ID.

    4. **Process Tool Result and Respond:**
       - For **success** status: **Present the information in a summarised and categorised manner providing key insights into the reviews and ratings.**
       - For **not_found** status: Inform the user that no data was found for that route ID
       - For **error** status: Acknowledge the error gracefully and suggest trying again

    5. **Handle Missing Route ID:** If the user doesn't provide a route ID, politely ask them to specify one.
    
    **Important Guidelines:**
    - Always extract the route ID before calling any tools
    - Don't call tools without a valid route ID
    - Provide clear, helpful responses based on the tool results
    - If the user asks for both reviews and ratings, you can call both tools sequentially
    - Once you've provided a complete response, don't re-prompt for the same information
    

    **Example Interactions:**
    - "What's the rating for bus route 123?" → Use `get_bus_rating_tool`
    - "Show me reviews for route 456" → Use `Bus_review_tool`
    - "Tell me about bus route 789" → Ask if they want reviews, ratings, or both
    """
)

# Set the agent as the root agent for standalone execution
root_agent = Bus_Review_Rating_Agent