import os
import logging
from typing import Dict, Any
import pandas as pd
import numpy as np
import faiss
from fastapi import FastAPI, Request, Response, HTTPException, Header, Depends
from fastapi.responses import JSONResponse
import uvicorn
from openai import OpenAI

from config import OPENAI_API_KEY, NAVERTALK_AUTH_TOKEN, EMBEDDINGS_FILE
# from config import NAVERTALK_VERIFY_TOKEN
from models import NaverTalkEvent, NaverTalkResponse, TextResponseContent
from chatbot import Chatbot

# Configure logging
logging.basicConfig(level = logging.INFO, format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize OpenAI client
client = OpenAI(api_key = OPENAI_API_KEY)

# Load data and embeddings
try:
    df = pd.read_csv(EMBEDDINGS_FILE)
    df['embedding'] = df['embedding'].apply(eval).apply(np.array)
    documents = df['text'].tolist()
    embeddings = df["embedding"].tolist()
    embedding_matrix = np.array(embeddings)
    index = faiss.IndexFlatL2(embedding_matrix.shape[1])
    index.add(embedding_matrix)
    logger.info(f"Data loaded successfully: {len(df)} entries")
except Exception as e:
    logger.error(f"Error loading data: {str(e)}")
    raise

# Define a system message
system_message = """
You are a Korean chatbot for a website called 비즈톡. Your primary role is to answer user questions 
using the information provided in the embedded text file.

Instructions:
1. Use only provided information for text-based queries.
2. If the answer is not in the embedded text file, politely state that you don't have that information.
3. Always respond in Korean.
4. Keep your answers concise and to the point.
"""

# Initialize FastAPI
app = FastAPI(title = "NaverTalk Chatbot API")


# Verify request is from NaverTalk
# async def verify_auth_token(authorization: str = Header(None)):
#     if authorization != f"ct_{NAVERTALK_AUTH_TOKEN}":
#         raise HTTPException(status_code = 401, detail = "Unauthorized")
#     return authorization

async def verify_auth_token(request: Request):
    """Verify authentication with better debugging"""
    headers = dict(request.headers)
    auth_header = headers.get("authorization")

    # Debug logging
    logger.info(f"All headers received: {headers}")
    logger.info(f"Authorization header: {auth_header}")

    # NaverTalk might be using a different header name or not sending auth
    # For testing, we'll accept requests without auth
    if not auth_header:
        logger.warning("No authorization header found - proceeding anyway for testing")
        return True

    expected_token = f"ct_{NAVERTALK_AUTH_TOKEN}"
    if auth_header != expected_token:
        logger.error(f"Token mismatch. Expected: {expected_token}, Got: {auth_header}")
        return False

    return True

# Chatbot instance
chatbot = Chatbot(index, embeddings, documents, system_message, client)

# User session storage
user_sessions = {}


@app.get("/")
async def root():
    return {"message": "NaverTalk Chatbot API is running"}


# @app.post("/webhook")
# async def handle_webhook(event: NaverTalkEvent, authorization: str = Depends(verify_auth_token)):
#     """Handle incoming webhook events from NaverTalk"""
#     logger.info(f"Received webhook event: {event.event}")
#
#     # Handle verification event
#     if event.event == "persistentMenu":
#         logger.info("Processing persistent menu event")
#         return JSONResponse(status_code = 200, content = {})
#
#     # Handle a user message
#     if event.event == "send" and event.textContent:
#         user_id = event.user
#         user_message = event.textContent.text
#
#         logger.info(f"User {user_id} sent message: {user_message}")
#
#         # Process message with chatbot
#         response_message = chatbot.chat(user_message)
#
#         # Create NaverTalk response
#         response = NaverTalkResponse(
#             user = user_id,
#             textContent = TextResponseContent(text = response_message)
#         )
#
#         logger.info(f"Sending response to user {user_id}: {response_message[:50]}...")
#         return response.model_dump()
#
#     # Return empty response for other events
#     return JSONResponse(status_code = 200, content = {})


'''
'''


@app.get("/webhook")
async def verify_webhook():
    """Handle GET requests for webhook verification"""
    logger.info("Received GET request to webhook - likely for verification")
    return JSONResponse(status_code=200, content={})


@app.post("/webhook")
async def handle_webhook(request: Request):
    """Handle incoming webhook events from NaverTalk with flexible authentication"""
    try:
        # Log raw request data
        body = await request.json()
        logger.info(f"Raw webhook data: {body}")

        # Verify authentication (but proceed even if it fails for now)
        auth_valid = await verify_auth_token(request)
        if not auth_valid:
            # For testing, we'll accept unauthenticated requests
            logger.warning("Proceeding with request despite authentication failure")

        # Parse event
        event_type = body.get("event")
        logger.info(f"Processing event type: {event_type}")

        # Handle verification event
        if event_type == "persistentMenu":
            logger.info("Processing persistent menu event")
            return JSONResponse(status_code=200, content={})

        # Handle user message
        if event_type == "send" and "textContent" in body:
            user_id = body.get("user")
            text_content = body.get("textContent", {})
            user_message = text_content.get("text", "")

            logger.info(f"User {user_id} sent message: {user_message}")

            try:
                # Process message with chatbot
                response_message = chatbot.chat(user_message)
                logger.info(f"Generated response: {response_message}")
            except Exception as e:
                logger.error(f"Error in chatbot processing: {str(e)}", exc_info=True)
                response_message = "죄송합니다. 메시지 처리 중 오류가 발생했습니다."

            # Create NaverTalk response format
            response = {
                "event": "send",
                "user": user_id,
                "textContent": {
                    "text": response_message
                }
            }

            logger.info(f"Sending response: {response}")
            return JSONResponse(status_code=200, content=response)

        # Handle unrecognized events
        logger.warning(f"Unhandled event type: {event_type}")
        return JSONResponse(status_code=200, content={})

    except Exception as e:
        logger.error(f"Webhook processing error: {str(e)}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": "Internal server error"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host = "0.0.0.0", port = port, reload = True)