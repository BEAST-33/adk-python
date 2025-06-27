from flask import Flask, request, jsonify
import requests
import datetime

app = Flask(__name__)

# Fixed passenger details
PASSENGER_NAME = "Sandip"
PASSENGER_EMAIL = "b.sandeep@gmail.com"
PASSENGER_MOBILE = "9876543210"

# Sample location mapping (replace with real data or API)
LOCATION_MAPPING = {
    "guna": {"BoardingPointId": 1765157, "DroppingPointId": 17773315, "LocationId": "1365", "LocationName": "Guna"},
    "dwarka": {"BoardingPointId": 1234567, "DroppingPointId": 17773315, "LocationId": "1136", "LocationName": "Dwarka"},
    # Add more mappings as needed
}

def get_location_info(location_name):
    key = location_name.strip().lower()
    return LOCATION_MAPPING.get(key, None)

@app.route('/', methods=['GET', 'POST'])
def agent_card():
    """
    Serve the Agent Card JSON at the root path.
    Always returns JSON response with Content-Type application/json.
    """
    agent_card_json = {
        "name": "Redbus Tentative Booking Agent",
        "description": "Helps users create tentative bus ticket bookings with dynamic source and destination inputs.",
        "url": "https://localhost:8033/webhook",  # Replace with your actual webhook URL
        "provider": {
            "organization": "Your Company Name",
            "url": "https://redbus.in"
        },
        "version": "1.0.0",
        "documentationUrl": "https://localhost:8033/docs",
        "capabilities": {
            "streaming": False,
            "pushNotifications": False,
            "stateTransitionHistory": False
        },
        "defaultInputModes": ["text/plain", "application/json"],
        "defaultOutputModes": ["text/plain", "application/json"],
        "skills": [
            {
                "id": "tentative-booking",
                "name": "Tentative Bus Ticket Booking",
                "description": "Creates tentative bus ticket bookings based on user-provided source and destination.",
                "tags": ["booking", "bus", "travel", "tickets"],
                "examples": [
                    "Book a bus from Delhi to Mumbai",
                    "Tentatively reserve a bus ticket from Bangalore to Chennai"
                ],
                "inputModes": ["text/plain", "application/json"],
                "outputModes": ["text/plain", "application/json"]
            }
        ]
    }

    return jsonify(agent_card_json)

@app.route('/webhook', methods=['POST'])
def webhook():
    """
    Handle the booking webhook from Google Assistant or other clients.
    """
    try:
        req = request.get_json(force=True)

        # Extract parameters from Dialogflow / Actions Builder request
        parameters = req.get('queryResult', {}).get('parameters', {})
        source = parameters.get('source')
        destination = parameters.get('destination')
        date_of_journey = parameters.get('date')  # ISO date string e.g. "2025-05-23"

        if not source or not destination:
            return jsonify({
                "fulfillmentText": "Please provide both source and destination locations to proceed with booking."
            })

        source_info = get_location_info(source)
        dest_info = get_location_info(destination)

        if not source_info or not dest_info:
            return jsonify({
                "fulfillmentText": f"Sorry, I could not find location details for '{source}' or '{destination}'. Please try again with different locations."
            })

        # Format date for API (e.g. "23-May-2025")
        if date_of_journey:
            try:
                dt = datetime.datetime.fromisoformat(date_of_journey)
                formatted_date = dt.strftime("%d-%b-%Y")
            except Exception:
                formatted_date = datetime.datetime.now().strftime("%d-%b-%Y")  # fallback to today
        else:
            formatted_date = datetime.datetime.now().strftime("%d-%b-%Y")  # default today

        # Construct payload dynamically
        payload = {
            "isInFunnel": True,
            "disableRedDeal": False,
            "IsAddOnSelected": False,
            "isBusPassInFunnel": False,
            "IsCovidOptIn": False,
            "isOpenTicket": False,
            "IsOptIn": False,
            "IsOptInForWhatsapp": True,
            "isRapAllowedTransaction": False,
            "isSeatlockOptin": False,
            "isShortRoute": False,
            "isSC": False,
            "isStreakAvailable": True,
            "isStreakOptin": False,
            "items": [
                {
                    "itemInfo": {
                        "oldTIN": "",
                        "journeyType": "ONWARD",
                        "oldItemUuid": "",
                        "SelectedCurrency": "INR",
                        "totalAmount": "",
                        "Trip": {
                            "BoardingPointId": source_info["BoardingPointId"],
                            "DateOfJourney": formatted_date,
                            "dropLocation": "",
                            "DroppingPointId": dest_info["DroppingPointId"],
                            "dstLocationId": dest_info["LocationId"],
                            "dstLocationName": dest_info["LocationName"],
                            "isFreeMealOpted": False,
                            "IsReturn": False,
                            "isbd": False,
                            "OperatorId": 15926,
                            "PassengerList": [
                                {
                                    "IsPrimaryPassenger": True,
                                    "PaxList": {
                                        "22": "Male",
                                        "1": "25",
                                        "201": "Tamil Nadu",
                                        "4": PASSENGER_NAME,
                                        "5": PASSENGER_EMAIL,
                                        "6": PASSENGER_MOBILE
                                    },
                                    "seatNumber": "10",
                                    "solarId": 14,
                                    "userInputLanguage": "en"
                                }
                            ],
                            "pickUpLocation": "",
                            "policyId": 0,
                            "RrouteId": 0,
                            "RouteId": 10225040,
                            "SelectedSeats": ["10"],
                            "srcLocationId": source_info["LocationId"],
                            "srcLocationName": source_info["LocationName"]
                        }
                    },
                    "itemType": "BUS",
                    "journeyType": "ONWARD"
                }
            ],
            "tags": ["NORMAL", "IND_RG_ELIG"]
        }

        # Headers from your curl command (adjust if needed)
        headers = {
            "PigeonDID": "",
            "DeviceId": "1a0d817e50aa6b23",
            "True-Client-IP": "106.216.225.144,163.116.214.42",
            "ga_client_id": "ad4e1e13-43d0-4d32-9c0f-6730e517ee56",
            "SourceApp": "",
            "ExpVariantName": "SRP_CLOSEST_BP_AB:V0,RAILS_PNR_V2:V2,RTC_TUPLE_CTR_ENHANCEMENT_EXP:V0,SC_RATING_NORATING:V2,SC_MINIMAL_CUSTINFO_V3_EXP:V0,RAILS_FC_SG_TRVL_RB:V2,BUSBUDDY_LOB_CROSS_SELL:V1,SC_SRP_GPS:V0,RAIL_SEARCH_BY_TRAIN_RB:V1,REDDEAL_BOOSTED_INV_AB:V1,Test_Color:\"00000\",IND_DARK_APPTHEME:V0,LOCATION_BASED_BP_SUGGESTION:V0,SC_CUSTINFO_EXP:V0,PERZ_HOME_CARDS_AB_ANDROID:V0,PERZ_SORT_V2:V0,VERTEX_AI_TRANSCRIPTION:V0,HOTEL_EXP:V0,CONVENIENCE_FEE_AB:V0,SC_USP_FTC:V0,SHORT_ROUTE_ANDROID:V3,RTC_INLINE_IMAGE_FILTER:V0,VERTEX_AI_TRANSCRIPTION_IND:V0,HOTEL_FEATURED_CARD_AB:V2,SC_SRP_NewUserCoupon:V0,HOTEL_LOB_TITLE:V1,SHORT_LMB_INLINE_FILTER:V2,RAILS_TRIP_REWARD:V1,FREEBIE_EXP_AB:V0,UNIFIED_SRP_V2:V4,RAILS_SRP_RB:V1,IND_PREF_BPDP_FILTER_iOS:V0,MINIMAL_CUSTINFO_V3_EXP:V0,CONENIENCE_FEE_AB:V0,RAILS_COMMON_HOME_WIDGET:V0,UPCOMING_TRIP_LTS_RB:V0,DARK_THEME_AB:V2,SHORT_ROUTE_LMB_AB:V0,HOTEL_ROOM_DETAIL_MERGE_EXP:V2,SC_NEWUSERCODE_SRP:V0,REDDEAL_BOOST_V2:V0,UNIFIED_SRP:V0,HOTEL_LAUNCH:V0,SEAT_LAYOUT_SEAT_IMAGES_AB:V0,UNIFIED_APP_TEST:V1,RAILS_HOME_PAGE_RB:V2,VISUAL_HOLDING_AB:V0,EARLY_BIRD_RED_DEAL:V0,RAIL_SEARCH_BY_TRAIN_FROM_SEARCH:V1,SHRINK_SEARCH_WIDGET_V2:V0,DARK_THEME_ENABLED:V0",
            "Language": "en",
            "Currency": "INR",
            "OSVersion": "15",
            "regid": "eT4aTGZKQaeFtG0n1hVZHF:APA91bFOzqz_VsPvgFPCt6SPGrhhjyZvAM3DN6gE1cin8TS3sZpyHkYQR-xr2s-DUASvLdIRFAR47DzEGW-I9eNIrE3_q4Bg9Jlmk9KZkeWMym1W023SKWc",
            "Country_Name": "IND",
            "UuidAtSRP": "73b2db259bc04df8839ca67109ebbdab",
            "SelectedCurrency": "INR",
            "Channel_Name": "MOBILE_APP",
            "Google_Aid": "ad4e1e13-43d0-4d32-9c0f-6730e517ee56",
            "os": "Android",
            "Accept": "application/json",
            "appversion": "80.2.0-IB2 Dev",
            "auth_key": "487a342c-92f1-41ae-81fa-aaa5120f6bb3",
            "ThirdPartySalesChannel": "",
            "AppVersionCode": "802000",
            "BusinessUnit": "BUS",
            "Country": "India",
            "AuthToken": "747dec22-9f28-4712-9af0-1f9c0a44adaa,914983eb-d6a6-453a-b896-ffa552ea1bf4,a6fdabca-b71c-4cee-b7ab-17c8f8130224,b12a7531-ff56-4de5-9c10-927434acee4b",
            "auth_token": "57:3B:F6:F7:E1:54:05:34:A5:64:8E:FF:D2:65:6C:44:B6:12:BB:DC",
            "UserType": "RETURNING",
            "MriSessionId": "AM4e6ac264-28b9-3c69-9c38-0594ddc90772_AM4e6ac264-28b9-3c69-9c38-0594ddc90772"
        }

        api_url = "https://capi.redbus.com/api/Order/v4/Create"
        response = requests.post(api_url, headers=headers, json=payload)
        response_data = response.json()

        if response.status_code == 200:
            order_id = response_data.get("orderId", "N/A")
            fulfillment_text = f"Your tentative booking from {source.title()} to {destination.title()} for {PASSENGER_NAME} is successful. Your order ID is {order_id}."
        else:
            fulfillment_text = f"Sorry, I was unable to complete your booking. Please try again later."

        return jsonify({"fulfillmentText": fulfillment_text})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"fulfillmentText": "Sorry, something went wrong while processing your booking."})

@app.route('/tasks/send', methods=['POST'])
def tasks_send():
    """
    Example endpoint to handle /tasks/send.
    You can customize this for task delegation or other purposes.
    """
    try:
        data = request.get_json(force=True)
        # For demo, just echo back received data with a confirmation message
        return jsonify({
            "status": "success",
            "message": "Task received and processed.",
            "receivedData": data
        })
    except Exception as e:
        print(f"Error in /tasks/send: {e}")
        return jsonify({
            "status": "error",
            "message": "Failed to process task."
        }), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
