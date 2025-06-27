import logging
import json
import urllib.parse
import http.client
from pydantic import BaseModel
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Helper Function for HTTP POST Requests ---
def make_post_request(url: str, payload: dict, headers: dict) -> dict | None:
    """
    Makes a POST request to the specified URL with the given payload and headers.
    Handles basic HTTP connection and JSON response parsing.
    """
    try:
        logger.debug(f"Making POST request to: {url}")
        logger.debug(f"Payload: {payload}")

        parsed_url = urllib.parse.urlparse(url)
        # Use HTTPSConnection if the scheme is 'https', otherwise HTTPConnection
        if parsed_url.scheme == 'https':
            conn = http.client.HTTPSConnection(parsed_url.hostname, parsed_url.port or 443)
        else:
            conn = http.client.HTTPConnection(parsed_url.hostname, parsed_url.port or 80)

        conn.request("POST", parsed_url.path, json.dumps(payload), headers)
        response = conn.getresponse()
        data = response.read().decode('utf-8')

        logger.debug(f"Response status: {response.status}")
        logger.debug(f"Response data: {data}")

        conn.close()
        return json.loads(data)
    except Exception as e:
        logger.error(f"Error making HTTP request to {url}: {e}")
        return None

# --- Define the Response Model for the Tool ---
class NomicEmbeddingResponse(BaseModel):
    """
    Represents the response structure for the generate_nomic_embedding_tool.
    """
    status: str  # e.g., "success", "error", "no_text"
    message: str # A descriptive message
    embedding: list[float] | None = None # The generated embedding as a list of floats

# --- Tool Function for ADK Agent ---
def generate_nomic_embedding_tool(text_to_embed: str) -> NomicEmbeddingResponse:
    """
    Generates Nomic embeddings for the given text using an external service.

    Args:
        text_to_embed: The input text for which to generate embeddings.

    Returns:
        NomicEmbeddingResponse: An object containing the status, message, and the embedding itself.
    """
    logger.info(f"generate_nomic_embedding_tool called for text (first 50 chars): '{text_to_embed[:50]}'")
    
    if not text_to_embed:
        logger.warning("No text provided for embedding.")
        return NomicEmbeddingResponse(
            status="no_text",
            message="No text was provided for embedding.",
            embedding=None
        )

    api_url = "http://10.166.8.126:11434/api/embed" # Default URL for the embedding service
    model_name = "nomic-embed-text:latest"         # Default model name
    
    payload = {"model": model_name, "input": text_to_embed}
    headers = {'Content-Type': 'application/json'}
    
    try:
        result = make_post_request(api_url, payload, headers)
        if not result:
            logger.error("Failed to get a response from the embedding service.")
            return NomicEmbeddingResponse(
                status="error",
                message="Failed to get a response from the embedding service.",
                embedding=None
            )
        
        embeddings = result.get("embeddings", []) # Nomic's API typically returns 'embeddings'
        
        if isinstance(embeddings, list) and len(embeddings) > 0 and isinstance(embeddings[0], list) and len(embeddings[0]) > 0:
            # The API returns a list of embeddings, take the first one
            generated_embedding = embeddings[0]
            logger.info(f"Embedding generated successfully. Length: {len(generated_embedding)}")
            return NomicEmbeddingResponse(
                status="success",
                message=f"Nomic embedding generated successfully. Length: {len(generated_embedding)}",
                embedding=generated_embedding
            )
        else:
            logger.error(f"Unexpected embedding format: {result}")
            return NomicEmbeddingResponse(
                status="error",
                message="Could not parse embedding from service response.",
                embedding=None
            )
    except Exception as e:
        logger.error(f"Error during embedding generation: {e}", exc_info=True)
        return NomicEmbeddingResponse(
            status="error",
            message=f"An error occurred during embedding generation: {str(e)}.",
            embedding=None
        )

# --- Create the LLM agent ---
NomicEmbeddingAgent = LlmAgent(
    name="Nomic_Embedding_Agent",
    model=LiteLlm(model="ollama_chat/llama3.2:latest"), # Ensure this model is accessible and configured correctly
    tools=[generate_nomic_embedding_tool], # Register the new tool
    instruction="""
    You are a Nomic embedding generation assistant. Your primary goal is to take a user's text input,
    generate its Nomic embedding, and then present the result clearly.

    **Strict Steps:**
    
    1.  **Extract Text:** Identify and extract the user's main textual input for which they want an embedding.
        If the user provides text like "Generate embedding for 'Hello world'", extract "Hello world".
    2.  **Call Tool:** Use the `generate_nomic_embedding_tool` function with the extracted text.
    3.  **Process Tool Result and Respond:**
        * If the tool returns a `status: "success"`, inform the user that the embedding was generated, mention its length (e.g., "Embedding generated with length 768."), and then print the full embedding as a JSON string (or a list representation).
          **Example:** "Here is the Nomic embedding for your text (length 768): [0.123, -0.456, ...]"
    *critical* output the embedding array with all its elements, dont look for any mathematical pattern, the output is just the whole array with all its
          
          After presenting the embedding, **your task for this turn is complete, and you should provide a final, helpful response to the user.**
        * If the tool returns a `status: "no_text"`, inform the user that no text was provided for embedding and politely ask them to provide some text. **Then, your task for this turn is complete.**
        * If the tool returns a `status: "error"`, acknowledge the error gracefully and suggest they try again later or with different text. **Then, your task for this turn is complete.**
    4.  **Handle Missing Text:** If the user does not provide any text to embed in their initial message, politely ask them to specify the text. Do NOT try to call the tool without input text.
    **DO not add any data analysis methods , just give the plain response without making up things on your own **
    **Important:** Once you have provided a response based on the `generate_nomic_embedding_tool`'s output (whether success, no_text, or error), consider the current turn's objective achieved. **Do not re-prompt for the same text or re-call the tool for the same request.** Focus on providing a single, conclusive answer for each user query.
    
    """
)

# Set the NomicEmbeddingAgent as the root agent for standalone execution
root_agent = NomicEmbeddingAgent
