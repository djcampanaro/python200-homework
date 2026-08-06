from dotenv import load_dotenv
from openai import OpenAI
from pprint import pprint

if load_dotenv():
    print("Successfully loaded api key")

client = OpenAI()

# --- Completions API ---

# API Q1


