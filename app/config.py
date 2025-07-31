import os
from dotenv import load_dotenv

# for local development only
load_dotenv()

# --- API Keys and Environment ---
# Multi-key configuration for 3x performance
GROQ_API_KEY_1 = os.getenv("GROQ_API_KEY_1")
GROQ_API_KEY_2 = os.getenv("GROQ_API_KEY_2") 
GROQ_API_KEY_3 = os.getenv("GROQ_API_KEY_3")

# Collect all valid keys
GROQ_API_KEYS = []
for i, key in enumerate([GROQ_API_KEY_1, GROQ_API_KEY_2, GROQ_API_KEY_3], 1):
    if key and len(key.strip()) > 0:
        GROQ_API_KEYS.append(key.strip())
        print(f"🔑 Loaded API Key #{i}: ...{key.strip()[-8:]}")
    else:
        print(f"⚠️ API Key #{i} not found or empty")

# Validate keys
if len(GROQ_API_KEYS) == 0:
    print("❌ ERROR: No valid Groq API keys found!")
    GROQ_API_KEY = None
else:
    print(f"✅ Total valid keys: {len(GROQ_API_KEYS)}")
    GROQ_API_KEY = GROQ_API_KEYS[0]  # Primary key for backward compatibility

# Rate limiting configuration per key
GROQ_RATE_LIMITS = {
    'requests_per_minute': 25,  # Per key limit (Conservative, Groq allows 30)
    'tokens_per_minute': 5500,  # Per key limit (Conservative, Groq allows 6000)
    'buffer_seconds': 2,        # Safety buffer
    'total_capacity': len(GROQ_API_KEYS) * 25,  # Total RPM capacity
}

# hardcoded bearer token
API_BEARER_TOKEN = "52399ce2b5dde000da221a0495f36cad39a5a362c2c823aa31268d6b5ad18c76"

# --- Model and Search Configuration ---
EMBEDDING_MODEL = 'BAAI/bge-small-en-v1.5'
LLM_MODEL = 'llama3-8b-8192'
PINECONE_INDEX_METRIC = 'cosine'
TOP_K_RESULTS = 5