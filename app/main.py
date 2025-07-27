# In app/main.py

import hashlib
import asyncio
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from . import config
from .models import HackRxRequest, HackRxResponse
from .document_processor import get_and_chunk_pdf
from .vector_store import create_or_get_index, embed_and_store, retrieve_context
from .llm_handler import generate_all_answers_at_once_async

# --- FastAPI App Initialization ---
app = FastAPI(
    title="HackRx",
    description="An LLM-powered system to query large documents for the HackRx hackathon.",
    version="1.0.0"
)

# --- CONFIGURATION ---
# Define the size of our question batches. 3 is a safe number.
QUESTION_BATCH_SIZE = 3

# --- Security Dependency ---
security_scheme = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security_scheme)):
    if credentials.scheme != "Bearer" or credentials.credentials != config.API_BEARER_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid or missing authorization token")
    return credentials

# --- API Endpoint (FINAL BATCHED LOGIC) ---
@app.post("/hackrx/run",
          response_model=HackRxResponse,
          dependencies=[Depends(verify_token)],
          tags=["Query Pipeline"])
async def run_query_pipeline(request: HackRxRequest):
    """
    This endpoint processes a document, stores it, and answers questions
    using an optimized BATCHED strategy to respect API token limits.
    """
    try:
        # 1. Get or Create ChromaDB Collection for caching
        url_str = str(request.documents)
        collection_name = f"hackrx-{hashlib.md5(url_str.encode()).hexdigest()}"
        collection = create_or_get_index(collection_name)

        # 2. Process document if it's new
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

        # --- BATCHING LOGIC STARTS HERE ---
        all_final_answers = []
        question_list = request.questions
        
        # Create batches of questions, e.g., [[q1,q2,q3], [q4,q5,q6], ...]
        question_batches = [
            question_list[i:i + QUESTION_BATCH_SIZE]
            for i in range(0, len(question_list), QUESTION_BATCH_SIZE)
        ]

        print(f"Split {len(question_list)} questions into {len(question_batches)} batches of size {QUESTION_BATCH_SIZE}.")

        # Process each batch
        for i, batch in enumerate(question_batches):
            print(f"Processing Batch {i+1}/{len(question_batches)}...")
            
            # 3. Gather context for the CURRENT BATCH of questions
            batch_context_chunks = set()
            for q in batch:
                context_for_question = retrieve_context(q, collection)
                batch_context_chunks.add(context_for_question)
            
            final_batch_context = " ".join(batch_context_chunks)

            # 4. Make one API call PER BATCH
            batch_answers = await generate_all_answers_at_once_async(batch, final_batch_context)
            
            # Add the answers from this batch to our final list
            all_final_answers.extend(batch_answers)
            print(f"Batch {i+1} complete.")

        # --- BATCHING LOGIC ENDS HERE ---

        return HackRxResponse(answers=all_final_answers)

    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")

@app.get("/", tags=["Health Check"])
def read_root():
    return {"status": "ok", "message": "Welcome to the HackRx Query System!"}