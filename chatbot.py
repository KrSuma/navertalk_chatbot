import os
import logging
from typing import List, Dict, Any, Union
import numpy as np
import faiss
from openai import OpenAI

logger = logging.getLogger(__name__)


class Chatbot:
    def __init__(self, index, embeddings, documents, system_message, client: OpenAI):
        self.index = index
        self.embeddings = embeddings
        self.documents = documents
        self.system_message = system_message
        self.client = client
        self.chat_history = []

    def get_embedding(self, text, model = "text-embedding-3-large"):
        text = text.replace("\n", " ")
        return self.client.embeddings.create(input = [text], model = model).data[0].embedding

    def find_similar_document(self, user_embedding):
        _, top_indices = self.index.search(np.array([user_embedding]), 1)
        top_index = top_indices[0][0]
        return self.documents[top_index]

    def chat(self, user_input: str) -> str:
        try:
            logger.info(f"Processing chat input: {user_input}")
            user_embedding = self.get_embedding(user_input)
            logger.info("Generated embedding")

            similar_document = self.find_similar_document(user_embedding)
            logger.info(f"Found similar document: {similar_document[:100]}...")

            system_message = self.system_message + " " + similar_document

            messages = [
                {"role": "system", "content": system_message},
            ]

            for message in self.chat_history:
                messages.append(message)
            messages.append({"role": "user", "content": user_input})

            logger.info(f"Sending {len(messages)} messages to OpenAI API")
            response = self.client.chat.completions.create(
                model = "gpt-4o-mini-2024-07-18",
                temperature = 0.3,
                messages = messages
            )
            assistant_message = response.choices[0].message.content
            logger.info(f"Received response: {assistant_message[:100]}...")

            self.add_to_history("user", user_input)
            self.add_to_history("assistant", assistant_message)
            return assistant_message
        except Exception as e:
            logger.error(f"Error in chat method: {str(e)}", exc_info = True)
            return "죄송합니다. 응답을 생성하는 동안 오류가 발생했습니다. 다시 시도해 주세요."

    def add_to_history(self, role: str, content: str) -> None:
        self.chat_history.append({"role": role, "content": content})

    def clear_history(self) -> None:
        self.chat_history.clear()
