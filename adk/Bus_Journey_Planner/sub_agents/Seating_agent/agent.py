import logging
import json
import requests
from datetime import datetime
from collections import defaultdict
from typing import Optional, Dict
from google.adk.agents import LlmAgent
from google.adk.models import Gemini
from google.adk.tools import FunctionTool
from google.adk.models.lite_llm import LiteLlm
from pydantic import BaseModel
class SeatingSearchResponse(BaseModel):
    """
    Response model for seating search results.
    
    Attributes:
        status: Status of the search operation (success, error, etc.)
        route_id: The bus route ID for which availability was checked
        journey_date: Date of the journey in YYYY-MM-DD format
        available_seats: List of available seats with details
        fare_groups: Dictionary of fare groups with counts and seat numbers
        available_seats_count: Total count of available seats
        total_seats: Total number of seats on the bus
        error: Any error message encountered during the operation
    """
    status: str
    route_id: str
    journey_date: str
    available_seats: list
    fare_groups: dict
    available_seats_count: int
    total_seats: Optional[int] = None
    error: Optional[str] = None

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def get_seat_availability(route_id: str, journey_date: Optional[str] = None) -> SeatingSearchResponse:
    """
    Fetch seat availability for a specific bus route.
    
    Args:
        route_id: The bus route ID to check availability for
        journey_date: Optional date in YYYY-MM-DD format. Defaults to today if not provided.
        
    Returns:
        String containing JSON formatted seat availability data or error message
    """
    try:
        # Validate route_id is numeric
        if not route_id.isdigit():
            return SeatingSearchResponse(
                status="error",
                route_id=route_id,
                journey_date=journey_date or datetime.today().strftime('%Y-%m-%d'),
                available_seats=[],
                fare_groups={},
                available_seats_count=0,
                total_seats=None,
                error=f"Invalid route ID '{route_id}'. Route ID must be numeric."
            )

        # Use today's date if not provided
        if not journey_date:
            journey_date = datetime.today().strftime('%Y-%m-%d')

        api_url = f"http://channels.omega.redbus.in:8001/IASPublic/getRealTimeUpdate/{route_id}/{journey_date}"
        
        all_seat_data = []
        available_seats_count = 0
        fare_availability_groups = defaultdict(lambda: {'count': 0, 'currency': 'INR', 'seat_numbers': []})
        total_seats_from_api = None
        error_message = None

        logger.info(f"Fetching seat availability from: {api_url}")
        response = requests.get(api_url, timeout=15)
        response.raise_for_status()
        data = response.json()

        # Get total seats
        total_seats_from_api = data.get('totalSeats')
        if total_seats_from_api is not None:
            try:
                total_seats_from_api = int(total_seats_from_api)
            except (ValueError, TypeError):
                error_message = f"Warning: 'totalSeats' field invalid: {total_seats_from_api}"
                total_seats_from_api = None
        else:
            error_message = "'totalSeats' field missing in API response."

        # Process seat status
        seat_status_list = data.get('seatStatus', [])
        if not isinstance(seat_status_list, list):
            error_message = "'seatStatus' field is not a list."
            seat_status_list = []
        elif not seat_status_list:
            error_message = "'seatStatus' list is empty."

        for seat_data in seat_status_list:
            if isinstance(seat_data, dict):
                seat_static = seat_data.get('seatStatic', {})
                st_volatile = seat_data.get('stVolatile', {})
                
                if isinstance(seat_static, dict) and isinstance(st_volatile, dict):
                    seat_number = seat_static.get('no') or st_volatile.get('no')
                    seat_availability = st_volatile.get('stAv')
                    fare_info = st_volatile.get('fare', {})
                    seat_type = fare_info.get('seatType')
                    seat_amount = fare_info.get('amount')
                    currency_type = fare_info.get('currencyType', 'INR')

                    seat_info = {
                        'seatNumber': seat_number,
                        'availabilityStatus': seat_availability,
                        'seatType': seat_type,
                        'amount': seat_amount,
                        'currency': currency_type
                    }
                    all_seat_data.append(seat_info)

                    if seat_availability == 'AVAILABLE':
                        available_seats_count += 1
                        if seat_amount is not None:
                            try:
                                fare_key = float(seat_amount)
                                fare_availability_groups[fare_key]['count'] += 1
                                fare_availability_groups[fare_key]['currency'] = currency_type
                                fare_availability_groups[fare_key]['seat_numbers'].append(seat_number)
                            except (ValueError, TypeError):
                                pass

        # Prepare response
        return SeatingSearchResponse(
            status='success',
            route_id=route_id,
            journey_date=journey_date,
            available_seats=all_seat_data,
            fare_groups=dict(fare_availability_groups),
            available_seats_count=available_seats_count,
            total_seats=total_seats_from_api,
            error=error_message
        )

    except requests.exceptions.RequestException as e:
        error_msg = f"API request failed: {str(e)}"
        logger.error(error_msg)
        return SeatingSearchResponse(
            status='error',
            route_id=route_id,
            journey_date=journey_date,
            available_seats=[],
            fare_groups={},
            available_seats_count=0,
            total_seats=None,
            error=error_msg
        )
    
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        logger.error(error_msg)
        return SeatingSearchResponse(
            status='error',
            route_id=route_id,
            journey_date=journey_date,
            available_seats=[],
            fare_groups={},
            available_seats_count=0,
            total_seats=None,
            error=error_msg
        )

# Create the seat availability tool


# Create the LLM agent
Seating_agent = LlmAgent(
    name="Seating_Agent",
    model=LiteLlm(model="ollama_chat/llama3.2:latest"),
    description=("An agent to check bus seat availability for specific routes using RouteID."),
    instruction="""
You are a helpful bus seat availability assistant. You help users check seat availability by:

1. Extracting the route ID from user queries
2. Using the get_seat_availability tool to fetch current seat information
3. Presenting the results in a clear, organized format

When showing results, highlight key information like:
- Total available seats
- Different seat types available
- Fare information for different seat categories
- Any relevant warnings or errors

If the route ID is not provided or invalid:
1. Ask the user to provide a valid route ID
2. Explain that the route ID should be a numeric value

For date-specific queries:
1. Accept dates in YYYY-MM-DD format
2. Default to today's date if not specified
3. Validate and explain if the date format is incorrect
    """,
    tools=[get_seat_availability],
)




