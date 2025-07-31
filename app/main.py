import hashlib
import asyncio
import itertools
import json
import time
from datetime import datetime
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import requests
import io

from . import config
from .models import HackRxRequest, HackRxResponse
from .document_processor import get_and_chunk_pdf, analyze_document_strategy
from .vector_store import create_or_get_index, embed_and_store, embed_and_store_async, retrieve_context
from .llm_handler import generate_answer_async, answer_direct_llm
from .api_manager import api_manager

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
        answer = await generate_answer_async(question, context)
    print(f"Generated answer for: '{question[:50]}...'")
    return answer

# Global cache dictionary (will reset on server restart)
ANSWER_CACHE = {}

def get_cache_key(question: str, collection_name: str) -> str:
    """Generate a unique cache key for question + document combination"""
    combined = f"{collection_name}:{question.strip().lower()}"
    return hashlib.md5(combined.encode()).hexdigest()

async def get_cached_answer_or_generate(question: str, collection, collection_name: str) -> str:
    """
    Check cache first, generate answer if not found
    Enhanced with smart API key management
    """
    cache_key = get_cache_key(question, collection_name)
    
    # Check if answer is already cached
    if cache_key in ANSWER_CACHE:
        print(f"📋 Using cached answer for: '{question[:50]}...'")
        return ANSWER_CACHE[cache_key]
    
    # Get best available API key
    key_info = api_manager.get_best_key()
    if not key_info:
        return "All API services are currently unavailable. Please try again later."
    
    key_index, api_key = key_info
    
    # Generate new answer with API key debugging
    loop = asyncio.get_running_loop()
    context = await loop.run_in_executor(None, retrieve_context, question, collection)
    
    
    
    answer = await generate_answer_async(question, context)
    
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
    start_time = time.time()
    
    try:
        url_str = str(request.documents)
        questions = request.questions
        
        print(f"🎯 HACKRX PROCESSING")
        print(f"📄 Document: {url_str}")
        print(f"❓ Questions: {len(questions)}")
        
        # Log the questions to terminal and file
        collection_name = f"hackrx-{hashlib.md5(url_str.encode()).hexdigest()}"
        log_questions_to_file(questions, url_str, collection_name)
        
        # Download and analyze document for intelligent routing
        print(f"📥 Downloading document for analysis...")
        with requests.get(url_str, timeout=30) as r:
            r.raise_for_status()
            pdf_bytes = r.content
        
        # INTELLIGENT ROUTING - This is the key addition!
        from .document_processor import analyze_document_strategy
        strategy, metadata = analyze_document_strategy(pdf_bytes, url_str)
        
        if strategy == "direct_llm":
            print(f"🚀 FAST TRACK: Direct LLM processing ({metadata['reason']})")
            print(f"📊 Document size: {metadata['size_mb']:.1f}MB")
            
            # ENHANCED: Extract document-specific context for accurate answers
            document_context = None
            document_type = metadata.get('reason', '')
            
            try:
                import fitz
                with fitz.open(stream=io.BytesIO(pdf_bytes), filetype="pdf") as doc:
                    if len(doc) > 0:
                        # Extract more comprehensive context based on document type
                        if 'constitution' in document_type or 'general_content' in document_type:
                            # For constitutional/legal documents, extract first 3 pages
                            context_pages = min(3, len(doc))
                            full_context = ""
                            
                            for page_num in range(context_pages):
                                page_text = doc[page_num].get_text("text")
                                full_context += page_text + "\n"
                            
                            # Extract key identifying information
                            document_context = extract_document_identity(full_context, metadata)
                            
                        elif 'high_page_count' in document_type:
                            # For large documents, sample strategically
                            document_context = extract_strategic_context(doc, metadata)
                        else:
                            # Default: first page
                            document_context = doc[0].get_text("text")[:2000]
                            
            except Exception as e:
                print(f"⚠️ Context extraction failed: {e}")
                document_context = None
            
            # Direct LLM processing with enhanced context
            all_final_answers = await answer_direct_llm(questions, document_context, metadata)
            
        else:
            print(f"🏥 STANDARD RAG: Processing insurance document ({metadata['size_mb']:.1f}MB)")
            
            # Your existing RAG pipeline
            collection = create_or_get_index(collection_name)

            if collection.count() == 0:
                print("Collection is new or empty. Processing document...")
                chunks = get_and_chunk_pdf(url_str)
                if not chunks:
                    raise HTTPException(status_code=400, detail="Failed to extract text from the document.")
                
                documents_to_store = [chunk['text'] for chunk in chunks]
                metadata_to_store = [chunk['metadata'] for chunk in chunks]
                ids_to_store = [f"{collection_name}_chunk_{i}" for i in range(len(documents_to_store))]
                
                await embed_and_store_async(
                    documents_to_store,
                    metadata_to_store,
                    ids_to_store,
                    collection
                )
                print("Document processing and storage complete.")
            else:
                print("Document already processed. Using existing collection.")

            # Generate answers using RAG
            tasks = [get_cached_answer_or_generate(q, collection, collection_name) for q in questions]
            print(f"Starting concurrent generation of {len(tasks)} answers...")
            all_final_answers = await asyncio.gather(*tasks)
            print("All answers generated.")
        
        processing_time = time.time() - start_time
        print(f"⏱️ Total processing: {processing_time:.2f}s using {strategy.upper()} strategy")
        print(f"🎉 Successfully processed {len(questions)} questions")
        
        return HackRxResponse(answers=all_final_answers)

    except HTTPException:
        raise
    except Exception as e:
        print(f"💥 Error in query pipeline: {e}")
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")


@app.get("/", tags=["Health Check"])
def read_root():
    return {"status": "ok", "message": "Welcome to the HackRx Query System!"}

@app.get("/api-status", tags=["Health Check"])
async def api_status():
    """Get current API key status"""
    await api_manager.health_check_all_keys()
    
    healthy_keys = api_manager.get_healthy_keys()
    blocked_keys = list(api_manager.blocked_keys)
    
    return {
        "total_keys": len(config.GROQ_API_KEYS),
        "healthy_keys": len(healthy_keys),
        "blocked_keys": len(blocked_keys),
        "status": "healthy" if len(healthy_keys) > 0 else "unhealthy",
        "key_details": [
            {
                "key_index": i,
                "key_suffix": f"...{key[-8:]}",
                "healthy": api_manager.key_status[i]['healthy'],
                "blocked": i in api_manager.blocked_keys,
                "usage_count": api_manager.key_usage_count[i],
                "error_count": api_manager.key_status[i]['error_count']
            }
            for i, key in enumerate(config.GROQ_API_KEYS)
        ]
    }

def extract_document_identity(text: str, metadata: dict) -> str:
    """
    Extract key identifying information from document text
    """
    # Clean and prepare text
    text_clean = text.replace('\n', ' ').replace('\r', ' ')
    text_clean = ' '.join(text_clean.split())  # Remove extra spaces
    
    # Extract first 3000 characters for analysis
    sample = text_clean[:3000]
    
    # Look for key identifying phrases
    identity_markers = []
    
    # Constitution detection
    if any(word in sample.lower() for word in ['constitution', 'constitutional']):
        # Look for country/region identifiers
        if 'india' in sample.lower() or 'bharat' in sample.lower():
            identity_markers.append("Constitution of India")
        elif 'united states' in sample.lower() or 'america' in sample.lower():
            identity_markers.append("Constitution of United States")
        else:
            identity_markers.append("Constitutional document")
    
    # Legal document detection
    if 'supreme court' in sample.lower():
        if 'india' in sample.lower():
            identity_markers.append("Supreme Court of India")
        else:
            identity_markers.append("Supreme Court document")
    
    # Add more context
    title_match = None
    lines = text_clean.split('\n')[:10]  # First 10 lines
    for line in lines:
        if len(line.strip()) > 20 and len(line.strip()) < 200:
            # Likely a title
            title_match = line.strip()
            break
    
    # Build context string
    context_parts = []
    if identity_markers:
        context_parts.append(f"Document Type: {', '.join(identity_markers)}")
    if title_match:
        context_parts.append(f"Title: {title_match}")
    
    context_parts.append(f"Content Sample: {sample[:1500]}...")
    
    return '\n'.join(context_parts)

def extract_strategic_context(doc, metadata: dict) -> str:
    """
    Extract strategic context from large documents
    """
    total_pages = len(doc)
    context_parts = []
    
    # Always include first page (title/intro)
    if total_pages > 0:
        first_page = doc[0].get_text("text")[:1000]
        context_parts.append(f"Document Start: {first_page}")
    
    # Include table of contents if available (usually page 2-5)
    for page_num in range(1, min(5, total_pages)):
        page_text = doc[page_num].get_text("text")
        if 'contents' in page_text.lower() or 'index' in page_text.lower():
            toc_text = page_text[:800]
            context_parts.append(f"Table of Contents: {toc_text}")
            break
    
    return '\n'.join(context_parts)