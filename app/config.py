import os
from dotenv import load_dotenv

# for local development only
load_dotenv()

# PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")


# --- API Keys and Environment ---
# GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# --- API Keys and Environment ---
# Store keys as a list
GROQ_API_KEYS = [
    os.getenv("GROQ_API_KEY_1"),
    os.getenv("GROQ_API_KEY_2"),
    os.getenv("GROQ_API_KEY_3"),
    os.getenv("GROQ_API_KEY_4"),
]

# CRITICAL IMPROVEMENT: Filter out any keys that are missing or empty
GROQ_API_KEYS = [key for key in GROQ_API_KEYS if key]


# PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")


# hardcoded bearer token
API_BEARER_TOKEN = "52399ce2b5dde000da221a0495f36cad39a5a362c2c823aa31268d6b5ad18c76"

# --- Model and Search Configuration ---
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
LLM_MODEL = 'llama3-8b-8192'
PINECONE_INDEX_METRIC = 'cosine'
TOP_K_RESULTS = 5