import hashlib
import asyncio
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from . import config
from .models import HackRxRequest, HackRxResponse
from .document_processor import get_and_chunk_pdf
# These imports now correctly point to your ChromaDB-powered vector_store functions
from .vector_store import create_or_get_index, embed_and_store, retrieve_context_async
from .llm_handler import generate_answer_async

# --- FastAPI App Initialization ---
app = FastAPI(
    title="HackRx",
    description="An LLM-powered system to query large documents for the HackRx hackathon.",
    version="1.0.0"
)

# --- Security Dependency ---
security_scheme = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security_scheme)):
    """A dependency to verify the bearer token."""
    if credentials.scheme != "Bearer" or credentials.credentials != config.API_BEARER_TOKEN:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing authorization token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials

# --- CONCURRENCY CONTROL ---
CONCURRENCY_SEMAPHORE = asyncio.Semaphore(3)


# --- Helper coroutine to process a single question ---
# I've renamed 'index' to 'collection' for clarity
async def process_single_question(question: str, collection) -> str:
    """This function handles the full async pipeline for one question."""
    async with CONCURRENCY_SEMAPHORE:
        context = await retrieve_context_async(question, collection)
        answer = await generate_answer_async(question, context)
        print(f"Generated answer for: '{question}'")
        return answer


# --- API Endpoint ---
@app.post("/hackrx/run",
          response_model=HackRxResponse,
          dependencies=[Depends(verify_token)],
          tags=["Query Pipeline"])
async def run_query_pipeline(request: HackRxRequest):
    """
    This endpoint processes a document from a URL, stores its contents in a
    vector database, and answers a list of questions concurrently.
    """
    try:
        # 1. Create a unique name for the ChromaDB collection from the URL.
        url_str = str(request.documents)
        collection_name = f"hackrx-{hashlib.md5(url_str.encode()).hexdigest()}"
        
        # 2. Get or Create ChromaDB Collection (this is now instant)
        # Renamed 'index' to 'collection'
        collection = create_or_get_index(collection_name)

        # 3. **CRITICAL FIX**: Check if the collection is empty using ChromaDB's .count() method.
        if collection.count() == 0:
            print("Collection is new or empty. Processing document...")
            chunks = get_and_chunk_pdf(url_str)
            if not chunks:
                 raise HTTPException(status_code=400, detail="Failed to extract text from the document.")
            # This step is synchronous and happens only for new documents
            embed_and_store(chunks, collection)
        else:
            print("Document already processed. Using existing collection.")

        # 4. Process all questions concurrently
        tasks = [process_single_question(q, collection) for q in request.questions]
        
        print("Starting concurrent generation of all answers...")
        final_answers = await asyncio.gather(*tasks)
        print("All answers generated.")

        # 5. Return the structured JSON response
        return HackRxResponse(answers=final_answers)

    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")

@app.get("/", tags=["Health Check"])
def read_root():
    """A simple health check endpoint."""
    return {"status": "ok", "message": "Welcome to the HackRx Query System!"}