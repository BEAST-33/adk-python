from google.adk.agents import SequentialAgent, LlmAgent
from google.adk.models.lite_llm import LiteLlm

# Import the tools from individual agents

from Bus_Journey_Planner.sub_agents.Route_agent.agent import  Route_agent
from Bus_Journey_Planner.sub_agents.Review_agent.agent import Review_agent
from Bus_Journey_Planner.sub_agents.Seating_agent.agent import Seating_agent
from pydantic import BaseModel
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



# Main Sequential Agent for Complete Journey Planning
Bus_Journey_Planner = SequentialAgent(
    name="Bus_Journey_Planner",
    sub_agents=[Route_agent, Review_agent, Seating_agent],
    description="""
A comprehensive bus journey planning assistant that helps users by:
1. Finding available bus routes between locations
2. Providing user reviews and ratings for the routes
3. Checking real-time seat availability
4. Combining all information to help users make informed decisions

The agent follows this workflow:
1. First finds all available routes for the requested journey
2. Then fetches user reviews for the found routes
3. Finally checks current seat availability
    """
)
root_agent = Bus_Journey_Planner


# # Async function to handle journey planning
# async def plan_journey(query: Input):
#     """
#     Plan a complete bus journey with routes, reviews, and seat availability.
    
#     Args:
#         query: User's journey planning query (e.g., "Find buses from Delhi to Mumbai")
        
#     Returns:
#         Complete journey planning information including routes, reviews, and seat availability
#     """
#     try:
#         current_output = Output()
#         final_result = None
        
#         logger.info(f"Processing query: {query.query}")
#         async for result in Bus_Journey_Planner.run_async(query):
#             logger.info(f"Received result: {result}")
#             if isinstance(result, dict):
#                 if 'routes_data' in result:
#                     current_output.routes_data = result['routes_data']
#                     logger.info("Updated routes_data")
#                 if 'reviews_data' in result:
#                     current_output.reviews_data = result['reviews_data']
#                     logger.info("Updated reviews_data")
#                 if 'seating_data' in result:
#                     current_output.seating_data = result['seating_data']
#                     logger.info("Updated seating_data")
#                 final_result = current_output
#             else:
#                 logger.warning(f"Unexpected result type: {type(result)}")
        
#         if final_result is None:
#             logger.error("No results returned from journey planner")
#             return BusJourneyPlannerError(
#                 status="error",
#                 message="No results returned from journey planner"
#             )
            
#         logger.info(f"Final Result: {final_result.dict()}")
            
#         return BusJourneyPlannerResponse(
#             status="success",
#             message="Journey planning completed successfully",
#             data=final_result
#         )
#     except Exception as e:
#         import traceback
#         error_details = traceback.format_exc()
#         logger.error(f"Error in plan_journey: {error_details}")
#         return BusJourneyPlannerError(
#             status="error",
#             message=str(e)
#         )

# # Example usage
# if __name__ == "__main__":
#     print("Bus Journey Planner initialized successfully!")
#     print("You can now plan your bus journey with complete information.")
    
#     # Example queries
#     example_queries = [
#         "Find buses from Delhi to Mumbai",
#         "Show me buses from Bangalore to Chennai with reviews",
#         "Plan my journey from Kolkata to Pune"
#     ]
    
#     import asyncio
    
#     async def test_queries():
#         for query in example_queries:
#             print(f"\n--- Query: {query} ---")
#             try:
#                 result = await plan_journey(Input(query=query))
#                 print(result)
#             except Exception as e:
#                 print(f"Error: {e}")
    
#     asyncio.run(test_queries())

