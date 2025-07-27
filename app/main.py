# In app/main.py

import hashlib
import asyncio
import itertools
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from . import config
from .models import HackRxRequest, HackRxResponse
from .document_processor import get_and_chunk_pdf
from .vector_store import create_or_get_index, embed_and_store, retrieve_context
# Import the new, simpler handler
from .llm_handler import generate_answer_async

# --- FastAPI App Initialization ---
app = FastAPI(
    title="HackRx",
    description="An LLM-powered system to query large documents for the HackRx hackathon.",
    version="1.0.0"
)

# --- CONCURRENCY CONTROL ---
# This semaphore controls how many concurrent API calls we make to Groq.
# A good starting point is 2-3 calls per available API key.
# With 4 keys, a limit of 8-10 is safe and very fast.
API_CONCURRENCY_LIMIT = 8
API_SEMAPHORE = asyncio.Semaphore(API_CONCURRENCY_LIMIT)

# --- Security Dependency ---
security_scheme = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security_scheme)):
    if credentials.scheme != "Bearer" or credentials.credentials != config.API_BEARER_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid or missing authorization token")
    return credentials

# --- Helper Coroutine for a SINGLE Question (The most reliable method) ---
async def process_single_question(question: str, collection, api_key: str) -> str:
    """
    This function handles the full, high-accuracy pipeline for one question.
    """
    # We still retrieve context first
    loop = asyncio.get_running_loop()
    context = await loop.run_in_executor(
        None, retrieve_context, question, collection
    )
    
    # We then wrap the single, simple API call in our semaphore
    async with API_SEMAPHORE:
        answer = await generate_answer_async(question, context, api_key)
    
    print(f"Generated answer for: '{question[:50]}...'")
    return answer

# --- API Endpoint ---
@app.post("/hackrx/run",
          response_model=HackRxResponse,
          dependencies=[Depends(verify_token)],
          tags=["Query Pipeline"])
async def run_query_pipeline(request: HackRxRequest):
    """
    This endpoint uses the final, most robust architecture: concurrent processing
    of single questions, each with a simple prompt, controlled by a semaphore.
    """
    api_keys = config.GROQ_API_KEYS
    if not api_keys:
        raise HTTPException(status_code=500, detail="No Groq API keys configured on the server.")

    try:
        url_str = str(request.documents)
        collection_name = f"hackrx-{hashlib.md5(url_str.encode()).hexdigest()}"
        collection = create_or_get_index(collection_name)

        if collection.count() == 0:
            print("Collection is new or empty. Processing document...")
            chunks = get_and_chunk_pdf(url_str)
            if not chunks:
                 raise HTTPException(status_code=400, detail="Failed to extract text from the document.")
            
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, embed_and_store, chunks, collection)
            print("Document processing and storage complete.")
        else:
            print("Document already processed. Using existing collection.")

        # --- High-Accuracy Concurrent Processing ---
        questions = request.questions
        
        # This cycles through your list of API keys for each question
        key_cycler = itertools.cycle(api_keys)

        # Create a list of tasks, one for each question
        tasks = [
            process_single_question(q, collection, next(key_cycler))
            for q in questions
        ]

        print(f"Starting concurrent generation of {len(tasks)} answers...")
        all_final_answers = await asyncio.gather(*tasks)
        print("All answers generated.")
        
        return HackRxResponse(answers=all_final_answers)

    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")

@app.get("/", tags=["Health Check"])
def read_root():
    return {"status": "ok", "message": "Welcome to the HackRx Query System!"}