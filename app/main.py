import hashlib
import asyncio
import itertools
import json
from datetime import datetime
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from . import config
from .models import HackRxRequest, HackRxResponse
from .document_processor import get_and_chunk_pdf
# We will now need to update embed_and_store, so we'll adjust the call to it.
from .vector_store import create_or_get_index, embed_and_store, retrieve_context
from .llm_handler import generate_answer_async

# --- FastAPI App Initialization ---
app = FastAPI(
    title="HackRx",
    description="An LLM-powered system to query large documents for the HackRx hackathon.",
    version="1.0.0"
)

# --- CONCURRENCY CONTROL ---
API_CONCURRENCY_LIMIT = 8 
API_SEMAPHORE = asyncio.Semaphore(API_CONCURRENCY_LIMIT)

# --- Security Dependency ---
security_scheme = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security_scheme)):
    if credentials.scheme != "Bearer" or credentials.credentials != config.API_BEARER_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid or missing authorization token")
    return credentials

# --- Helper Coroutine for a SINGLE Question (No changes needed here) ---
async def process_single_question(question: str, collection, api_key: str) -> str:
    loop = asyncio.get_running_loop()
    context = await loop.run_in_executor(
        None, retrieve_context, question, collection
    )
    async with API_SEMAPHORE:
        answer = await generate_answer_async(question, context, api_key)
    print(f"Generated answer for: '{question[:50]}...'")
    return answer

# Add this to main.py after your existing imports
# Global cache dictionary (will reset on server restart)
ANSWER_CACHE = {}

def get_cache_key(question: str, collection_name: str) -> str:
    """Generate a unique cache key for question + document combination"""
    combined = f"{collection_name}:{question.strip().lower()}"
    return hashlib.md5(combined.encode()).hexdigest()

async def get_cached_answer_or_generate(question: str, collection, api_key: str, collection_name: str) -> str:
    """
    Check cache first, generate answer if not found
    """
    cache_key = get_cache_key(question, collection_name)
    
    # Check if answer is already cached
    if cache_key in ANSWER_CACHE:
        print(f"📋 Using cached answer for: '{question[:50]}...'")
        return ANSWER_CACHE[cache_key]
    
    # Generate new answer
    loop = asyncio.get_running_loop()
    context = await loop.run_in_executor(None, retrieve_context, question, collection)
    
    async with API_SEMAPHORE:
        answer = await generate_answer_async(question, context, api_key)
    
    # Cache the answer
    ANSWER_CACHE[cache_key] = answer
    print(f"💾 Cached new answer for: '{question[:50]}...'")
    
    return answer

# --- Log Questions to File ---
def log_questions_to_file(questions: list, document_url: str, collection_name: str):
    """
    Logs questions to a file for caching and analysis purposes.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    log_entry = {
        "timestamp": timestamp,
        "document_url": str(document_url),
        "collection_name": collection_name,
        "questions": questions,
        "question_count": len(questions)
    }
    
    # Print to terminal for immediate visibility
    print(f"\n{'='*60}")
    print(f"📝 QUESTIONS RECEIVED - {timestamp}")
    print(f"📄 Document: {document_url}")
    print(f"🔍 Collection: {collection_name}")
    print(f"❓ Total Questions: {len(questions)}")
    print(f"{'='*60}")
    
    for i, question in enumerate(questions, 1):
        print(f"{i:2d}. {question}")
    
    print(f"{'='*60}\n")
    
    # Also save to a JSON file for persistence
    try:
        log_filename = "questions_log.json"
        
        # Try to load existing log
        try:
            with open(log_filename, 'r', encoding='utf-8') as f:
                existing_logs = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            existing_logs = []
        
        # Append new entry
        existing_logs.append(log_entry)
        
        # Save updated log
        with open(log_filename, 'w', encoding='utf-8') as f:
            json.dump(existing_logs, f, indent=2, ensure_ascii=False)
            
        print(f"📁 Questions logged to: {log_filename}")
        
    except Exception as e:
        print(f"⚠️ Warning: Could not save questions to file: {e}")

def get_questions_for_document(document_url: str) -> list:
    """
    Retrieves all previously asked questions for a specific document.
    """
    try:
        with open("questions_log.json", 'r', encoding='utf-8') as f:
            logs = json.load(f)
        
        questions_for_doc = []
        for log_entry in logs:
            if log_entry.get("document_url") == document_url:
                questions_for_doc.extend(log_entry.get("questions", []))
        
        # Remove duplicates while preserving order
        unique_questions = list(dict.fromkeys(questions_for_doc))
        
        return unique_questions
        
    except (FileNotFoundError, json.JSONDecodeError):
        return []

def print_cached_questions(document_url: str):
    """
    Prints all cached questions for a document.
    """
    questions = get_questions_for_document(document_url)
    
    if questions:
        print(f"\n📋 CACHED QUESTIONS FOR: {document_url}")
        print(f"{'='*60}")
        for i, question in enumerate(questions, 1):
            print(f"{i:2d}. {question}")
        print(f"{'='*60}")
        print(f"Total unique questions: {len(questions)}\n")
    else:
        print(f"No cached questions found for: {document_url}")

# --- API Endpoint ---
@app.post("/hackrx/run",
          response_model=HackRxResponse,
          dependencies=[Depends(verify_token)],
          tags=["Query Pipeline"])
async def run_query_pipeline(request: HackRxRequest):
    api_keys = config.GROQ_API_KEYS
    if not api_keys:
        raise HTTPException(status_code=500, detail="No Groq API keys configured on the server.")

    try:
        url_str = str(request.documents)
        collection_name = f"hackrx-{hashlib.md5(url_str.encode()).hexdigest()}"
        
        # Log the questions to terminal and file
        log_questions_to_file(request.questions, request.documents, collection_name)
        
        collection = create_or_get_index(collection_name)

        if collection.count() == 0:
            print("Collection is new or empty. Processing document...")
            chunks = get_and_chunk_pdf(url_str)
            if not chunks:
                raise HTTPException(status_code=400, detail="Failed to extract text from the document.")
            
            documents_to_store = [chunk['text'] for chunk in chunks]
            metadata_to_store = [chunk['metadata'] for chunk in chunks]
            ids_to_store = [f"{collection_name}_chunk_{i}" for i in range(len(documents_to_store))]
            
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None, 
                embed_and_store, 
                documents_to_store,
                metadata_to_store,
                ids_to_store,
                collection
            )
            print("Document processing and storage complete.")
        else:
            print("Document already processed. Using existing collection.")

        questions = request.questions
        key_cycler = itertools.cycle(api_keys)
        # tasks = [
        #     process_single_question(q, collection, next(key_cycler))
        #     for q in questions
        # ]
        tasks = [
            get_cached_answer_or_generate(q, collection, next(key_cycler), collection_name)
            for q in questions
        ]

        print(f"Starting concurrent generation of {len(tasks)} answers...")
        all_final_answers = await asyncio.gather(*tasks)
        print("All answers generated.")
        
        return HackRxResponse(answers=all_final_answers)

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in query pipeline: {e}")
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")

@app.get("/", tags=["Health Check"])
def read_root():
    return {"status": "ok", "message": "Welcome to the HackRx Query System!"}