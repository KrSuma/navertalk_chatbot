import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# NaverTalk API settings
NAVERTALK_API_URL = "https://gw.talk.naver.com/chatbot/v1/event"
NAVERTALK_AUTH_TOKEN = os.getenv("NAVERTALK_AUTH_TOKEN")
# VERIFY_TOKEN = os.getenv("NAVERTALK_VERIFY_TOKEN")

# OpenAI settings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Data settings
# INPUT_FILE = os.getenv("INPUT_FILE", "dataset.csv")
EMBEDDINGS_FILE = os.getenv("EMBEDDINGS_FILE", "dataset_output.csv")
