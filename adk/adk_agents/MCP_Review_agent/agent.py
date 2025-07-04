# bus_review_mcp_agent.py
import os
from google.adk.agents import LlmAgent
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, StdioServerParameters
from google.adk.models.lite_llm import LiteLlm

# IMPORTANT: Replace this with the ABSOLUTE path to your bus_review_mcp_server.py script
PATH_TO_YOUR_MCP_SERVER_SCRIPT = "/Users/karthik.mr/adk-python/adk/adk_agents/MCP_Review_agent/bus_review_mcp_server.py"  # <<< REPLACE WITH ACTUAL PATH

if PATH_TO_YOUR_MCP_SERVER_SCRIPT == "/path/to/your/bus_review_mcp_server.py":
    print("WARNING: PATH_TO_YOUR_MCP_SERVER_SCRIPT is not set. Please update it in bus_review_mcp_agent.py.")
    # Optionally, raise an error if the path is critical
    # raise ValueError("Please set the correct path to your MCP server script")

# Create the LLM agent that uses the MCP server
Review_agent = LlmAgent(
    name="Review_Agent_MCP",
    model="gemini-2.0-flash", 
    tools=[
        MCPToolset(
            connection_params=StdioServerParameters(
                command='python3',  # Command to run your MCP server script
                args=[PATH_TO_YOUR_MCP_SERVER_SCRIPT],  # Argument is the path to the script
            ),
            # tool_filter=['Bus_review_tool']  # Optional: ensure only specific tools are loaded
        )
    ],
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

    The tool is now exposed via MCP server, so it will be available as `Bus_review_tool` in your toolset.
    """
)

# Set the Review_agent as the root agent for standalone execution
root_agent = Review_agent

