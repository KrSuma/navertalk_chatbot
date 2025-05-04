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

from config import OPENAI_API_KEY, VERIFY_TOKEN, NAVERTALK_AUTH_TOKEN, EMBEDDINGS_FILE
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
You are a Korean chatbot for a website called NaverTalk. Your primary role is to answer user questions 
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
async def verify_auth_token(authorization: str = Header(None)):
    if authorization != f"ct_{NAVERTALK_AUTH_TOKEN}":
        raise HTTPException(status_code = 401, detail = "Unauthorized")
    return authorization


# Chatbot instance
chatbot = Chatbot(index, embeddings, documents, system_message, client)

# User session storage
user_sessions = {}


@app.get("/")
async def root():
    return {"message": "NaverTalk Chatbot API is running"}


@app.post("/webhook")
async def handle_webhook(event: NaverTalkEvent, authorization: str = Depends(verify_auth_token)):
    """Handle incoming webhook events from NaverTalk"""
    logger.info(f"Received webhook event: {event.event}")

    # Handle verification event
    if event.event == "persistentMenu":
        logger.info("Processing persistent menu event")
        return JSONResponse(status_code = 200, content = {})

    # Handle a user message
    if event.event == "send" and event.textContent:
        user_id = event.user
        user_message = event.textContent.text

        logger.info(f"User {user_id} sent message: {user_message}")

        # Process message with chatbot
        response_message = chatbot.chat(user_message)

        # Create NaverTalk response
        response = NaverTalkResponse(
            user = user_id,
            textContent = TextResponseContent(text = response_message)
        )

        logger.info(f"Sending response to user {user_id}: {response_message[:50]}...")
        return response.model_dump()

    # Return empty response for other events
    return JSONResponse(status_code = 200, content = {})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host = "0.0.0.0", port = port, reload = True)