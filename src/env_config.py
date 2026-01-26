import os
from dotenv import load_dotenv

load_dotenv()

EMBEDDING_MODEL_TOKEN = os.getenv("EMBEDDING_MODEL_TOKEN_HF")
LLM = os.getenv("LLM")


