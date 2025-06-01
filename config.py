from dotenv import load_dotenv
import os

load_dotenv()

LLM_BASE_URL = os.getenv("LLM_BASE_URL")
LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_NAME = os.getenv("LLM_NAME")

