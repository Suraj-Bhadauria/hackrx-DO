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
QUESTION_BATCH_SIZE = 3

# --- Security Dependency ---
security_scheme = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security_scheme)):
    if credentials.scheme != "Bearer" or credentials.credentials != config.API_BEARER_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid or missing authorization token")
    return credentials

# --- Helper Coroutine for a Single Batch ---
async def process_question_batch(batch: list[str], collection) -> list[str]:
    """
    This function handles the logic for one batch of questions:
    1. Gathers context for the batch.
    2. Makes a single API call to the LLM.
    """
    print(f"Processing batch with {len(batch)} questions...")
    
    # Gather context for the CURRENT BATCH of questions
    batch_context_chunks = set()
    for q in batch:
        # retrieve_context is synchronous, so we run it in an executor
        # to avoid blocking the main async loop while others run.
        loop = asyncio.get_running_loop()
        context_for_question = await loop.run_in_executor(
            None, retrieve_context, q, collection
        )
        batch_context_chunks.add(context_for_question)
    
    final_batch_context = " ".join(batch_context_chunks)

    # Make one API call PER BATCH
    batch_answers = await generate_all_answers_at_once_async(batch, final_batch_context)
    print(f"Finished batch with {len(batch)} questions.")
    return batch_answers

# --- API Endpoint (FINAL CONCURRENT BATCHED LOGIC) ---
@app.post("/hackrx/run",
          response_model=HackRxResponse,
          dependencies=[Depends(verify_token)],
          tags=["Query Pipeline"])
async def run_query_pipeline(request: HackRxRequest):
    """
    This endpoint processes a document and answers questions using a
    CONCURRENT BATCHED strategy for maximum throughput.
    """
    try:
        # 1. Get or Create ChromaDB Collection
        url_str = str(request.documents)
        collection_name = f"hackrx-{hashlib.md5(url_str.encode()).hexdigest()}"
        collection = create_or_get_index(collection_name)

        # 2. Process document if it's new (The "expensive" first run)
        if collection.count() == 0:
            print("Collection is new or empty. Processing document...")
            chunks = get_and_chunk_pdf(url_str)
            if not chunks:
                 raise HTTPException(status_code=400, detail="Failed to extract text from the document.")
            
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, embed_and_store, chunks, collection)
            print("Document processing and storage complete.")
        else:
            # This case won't happen in the hackathon testing, but is good practice.
            print("Document already processed. Using existing collection.")

        # --- CONCURRENT BATCHING LOGIC STARTS HERE ---
        question_list = request.questions
        
        # Create batches of questions
        question_batches = [
            question_list[i:i + QUESTION_BATCH_SIZE]
            for i in range(0, len(question_list), QUESTION_BATCH_SIZE)
        ]

        print(f"Split {len(question_list)} questions into {len(question_batches)} batches to run concurrently.")

        # Create a list of tasks, one for each batch
        tasks = [process_question_batch(batch, collection) for batch in question_batches]
        
        # Run all batch tasks concurrently
        print("Starting concurrent processing of all batches...")
        results_from_batches = await asyncio.gather(*tasks)
        print("All batches have completed.")
        
        # The answers will be in a list of lists, so we need to flatten it
        # while preserving the original order of the questions.
        # We can do this by creating a map of question to answer.
        answer_map = {}
        for batch, answer_batch in zip(question_batches, results_from_batches):
            for question, answer in zip(batch, answer_batch):
                answer_map[question] = answer

        # Reconstruct the final answer list in the original order
        all_final_answers = [answer_map[q] for q in question_list]
        # --- CONCURRENT BATCHING LOGIC ENDS HERE ---

        return HackRxResponse(answers=all_final_answers)

    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")

@app.get("/", tags=["Health Check"])
def read_root():
    return {"status": "ok", "message": "Welcome to the HackRx Query System!"}