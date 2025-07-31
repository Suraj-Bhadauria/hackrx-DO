import requests
import io
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from collections import defaultdict
from transformers import AutoTokenizer
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import time
import os
import re
from typing import Tuple

# PRE-LOAD the tokenizer ONCE in the main process
print("🔄 Loading tokenizer in main process...")
GLOBAL_TOKENIZER = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
print("✅ Tokenizer loaded successfully!")

def num_tokens(text: str) -> int:
    """Helper function to count tokens for the splitter."""
    return len(GLOBAL_TOKENIZER.encode(text, add_special_tokens=False))

def process_single_page(page_data):
    """
    Process a single page in parallel - this function will run in separate processes
    """
    try:
        page_num, page_text, repeated_lines, chunk_size, chunk_overlap = page_data
        
        # DON'T recreate tokenizer - use a simple token counting approach
        def process_num_tokens(text: str) -> int:
            # Simple approximation: 1 token ≈ 4 characters for English text
            return len(text) // 4
        
        # Create text splitter for this process
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ". ", " ", ""],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=process_num_tokens,  # Use simple approximation
            is_separator_regex=False,
        )
        
        # Clean the page text by removing headers/footers
        lines = page_text.splitlines()
        clean_lines = [line for line in lines if line.strip() not in repeated_lines]
        clean_page_text = "\n".join(clean_lines).strip()
        
        # Skip empty pages
        if not clean_page_text:
            return []
        
        # Split the cleaned text
        subchunks = text_splitter.split_text(clean_page_text)
        
        # Create chunks with metadata
        page_chunks = []
        for i, chunk_text in enumerate(subchunks):
            page_chunks.append({
                "text": chunk_text,
                "metadata": {
                    "page": page_num,
                    "chunk_index": i,
                }
            })
        
        return page_chunks
        
    except Exception as e:
        print(f"Error processing page {page_num}: {e}")
        return []

def get_and_chunk_pdf(url: str) -> list[dict]:
    """
    TURBO MODE: Downloads and chunks PDF with parallel page processing
    """
    start_time = time.time()
    print(f"🚀 TURBO CHUNKING: Starting parallel processing for {url}")
    
    # Step 1: Download PDF
    print(f"📥 Downloading document...")
    download_start = time.time()
    try:
        with requests.get(url, stream=True, timeout=30) as r:
            r.raise_for_status()
            pdf_file_bytes = io.BytesIO(r.content)
    except requests.exceptions.RequestException as e:
        print(f"❌ Error downloading PDF: {e}")
        raise
    
    download_time = time.time() - download_start
    print(f"✅ Download completed in {download_time:.2f}s")

    # Step 2: Header/Footer Analysis
    print("🔍 Analyzing for headers/footers...")
    analysis_start = time.time()
    repeated_line_counts = defaultdict(int)
    total_pages = 0
    
    with fitz.open(stream=pdf_file_bytes, filetype="pdf") as doc:
        total_pages = len(doc)
        if total_pages == 0:
            print("⚠️ The PDF document is empty.")
            return []
            
        for page in doc:
            page_text = page.get_text("text")
            lines_on_page = {line.strip() for line in page_text.splitlines() if line.strip()}
            for line in lines_on_page:
                repeated_line_counts[line] += 1

    repeated_lines = {
        line for line, count in repeated_line_counts.items()
        if count == total_pages and len(line) > 10
    }
    
    analysis_time = time.time() - analysis_start
    print(f"✅ Analysis completed in {analysis_time:.2f}s")

    # Step 3: Configure chunking parameters
    chunk_size, chunk_overlap = 350, 75  # Default for small documents
    if total_pages > 100:  # Large document
        chunk_size = 500
        chunk_overlap = 100
        print(f"📊 Large document detected ({total_pages} pages) - using optimized chunking")
    elif total_pages > 50:   # Medium document
        chunk_size = 400

    # Step 4: PARALLEL PAGE PROCESSING
    print(f"⚡ Starting parallel processing of {total_pages} pages...")
    processing_start = time.time()
    
    # Prepare data for parallel processing
    page_data_list = []
    with fitz.open(stream=pdf_file_bytes, filetype="pdf") as doc:
        for page_num, page in enumerate(doc, start=1):
            page_text = page.get_text("text")
            page_data_list.append((page_num, page_text, repeated_lines, chunk_size, chunk_overlap))
    
    # Determine optimal number of workers
    max_workers = min(mp.cpu_count(), total_pages, 8)  # Cap at 8 to avoid overhead
    print(f"🔥 Using {max_workers} parallel workers for {total_pages} pages")
    
    all_chunks = []
    
    if total_pages > 10:  # Use parallel processing for larger documents
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Process pages in parallel
            page_chunks_list = list(executor.map(process_single_page, page_data_list))
            
            # Flatten results and add source URL
            for page_chunks in page_chunks_list:
                for chunk in page_chunks:
                    chunk["metadata"]["source"] = url
                    all_chunks.append(chunk)
    else:
        # For small documents, use sequential processing (less overhead)
        print("📄 Small document - using sequential processing")
        for page_data in page_data_list:
            page_chunks = process_single_page(page_data)
            for chunk in page_chunks:
                chunk["metadata"]["source"] = url
                all_chunks.append(chunk)
    
    processing_time = time.time() - processing_start
    total_time = time.time() - start_time
    
    if not all_chunks:
        print("⚠️ No usable chunks were created from the document.")
        return []
    
    # Performance metrics
    pages_per_second = total_pages / processing_time if processing_time > 0 else 0
    chunks_per_second = len(all_chunks) / total_time if total_time > 0 else 0
    
    print(f"🎉 TURBO CHUNKING COMPLETE!")
    print(f"📊 Processing: {processing_time:.2f}s ({pages_per_second:.1f} pages/sec)")
    print(f"⏱️ Total time: {total_time:.2f}s ({chunks_per_second:.1f} chunks/sec)")
    print(f"📝 Created {len(all_chunks)} semantic chunks with metadata")
    
    return all_chunks

# Add these detection patterns for known large documents
KNOWN_LARGE_DOCUMENTS = {
    'constitution': ['constitution', 'article', 'amendment', 'fundamental rights'],
    'principia': ['principia', 'newton', 'mathematics', 'theorem', 'physics'],
    'textbooks': ['chapter', 'section', 'exercise', 'bibliography'],
}

def analyze_document_strategy(pdf_bytes: bytes, url: str = None) -> tuple:
    """
    INTELLIGENT ROUTER: Determines the best processing strategy
    Enhanced with FIXED decision logic
    Returns: (strategy, metadata)
    """
    file_size = len(pdf_bytes)
    size_mb = file_size / (1024 * 1024)
    
    print(f"🔍 Document Analysis: {size_mb:.1f}MB")
    
    # Strategy 1: Size-based routing (PRIMARY FILTER)
    if size_mb > 10:  # Large documents
        print(f"📚 LARGE DOCUMENT DETECTED ({size_mb:.1f}MB) - Routing to Direct LLM")
        return "direct_llm", {"size_mb": size_mb, "reason": "large_document"}
    
    # Strategy 2: ENHANCED content analysis for medium AND smaller documents (>2MB)
    if size_mb > 2:  # Lowered threshold from 5MB to 2MB
        print(f"📄 Document ({size_mb:.1f}MB) - Analyzing content for type detection...")
        
        try:
            with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
                total_pages = len(doc)
                print(f"📊 Document has {total_pages} pages")
                
                # Enhanced page sampling - sample more pages for better detection
                pages_to_sample = min(5, total_pages)  # Sample first 5 pages
                sample_text = ""
                
                for page_num in range(pages_to_sample):
                    page_text = doc[page_num].get_text("text")
                    sample_text += page_text[:2000]  # Increased from 1000 to 2000 chars per page
                
                print(f"🔍 Sampled {len(sample_text)} characters from first {pages_to_sample} pages")
                
                # Enhanced keyword detection
                insurance_keywords = [
                    'policy', 'insurance', 'premium', 'coverage', 'claim', 'claims',
                    'deductible', 'beneficiary', 'policyholder', 'insured', 'insurer',
                    'liability', 'indemnity', 'underwriter', 'underwriting',
                    'motor insurance', 'health insurance', 'life insurance',
                    'sum assured', 'exclusion', 'inclusion', 'rider'
                ]
                
                # Enhanced general document indicators
                general_indicators = [
                    # Constitution-specific
                    'constitution', 'constitutional', 'article', 'amendment', 'fundamental rights',
                    'supreme court', 'high court', 'parliament', 'legislature', 'judiciary',
                    'union list', 'state list', 'concurrent list', 'directive principles',
                    'preamble', 'part i', 'part ii', 'part iii', 'part iv',
                    
                    # Academic/Scientific
                    'principia', 'mathematics', 'theorem', 'physics', 'newton',
                    'chapter', 'section', 'exercise', 'bibliography', 'references',
                    'university', 'academic', 'research', 'study', 'analysis',
                    
                    # Legal documents (non-insurance)
                    'criminal law', 'civil law', 'contract law', 'tort', 'statute',
                    'legal', 'court', 'judge', 'lawyer', 'attorney',
                    
                    # Technical manuals
                    'manual', 'specification', 'technical', 'engineering', 'software'
                ]
                
                # Count keyword occurrences
                sample_lower = sample_text.lower()
                insurance_score = sum(1 for keyword in insurance_keywords 
                                    if keyword in sample_lower)
                general_score = sum(1 for indicator in general_indicators 
                                  if indicator in sample_lower)
                
                print(f"📈 Keyword Analysis: Insurance={insurance_score}, General={general_score}")
                
                # FIXED DECISION LOGIC - Compare scores properly
                is_general_document = False
                reason = ""
                
                # Check 1: High page count usually indicates general documents
                if total_pages > 200:
                    print(f"🔍 High page count detected: {total_pages} pages")
                    # BUT check if it's still insurance-heavy
                    if insurance_score >= 5:  # Strong insurance indicators override page count
                        print(f"🏥 But strong insurance indicators ({insurance_score}) - keeping as insurance")
                        is_general_document = False
                    else:
                        is_general_document = True
                        reason = "high_page_count"
                
                # Check 2: FIXED - Strong general indicators BUT only if they dominate insurance
                elif general_score >= 3 and general_score > insurance_score:  # FIXED: Must be greater than insurance
                    print(f"🔍 General content dominates insurance content ({general_score} > {insurance_score})")
                    is_general_document = True
                    reason = "general_content"
                
                # Check 3: FIXED - Very low insurance score with some general indicators
                elif insurance_score <= 1 and general_score >= 2:  # FIXED: More restrictive
                    print(f"🔍 Very low insurance relevance with general indicators")
                    is_general_document = True
                    reason = "low_insurance_relevance"
                
                # Check 4: Specific document type detection from URL or content
                elif url and any(term in url.lower() for term in ['constitution', 'principia', 'manual', 'textbook']):
                    print(f"🔍 General document detected from URL pattern")
                    # BUT check if it's still insurance-heavy
                    if insurance_score >= 3:
                        print(f"🏥 But insurance indicators present ({insurance_score}) - keeping as insurance")
                        is_general_document = False
                    else:
                        is_general_document = True
                        reason = "url_pattern"
                
                # Check 5: FIXED - Large documents with general dominance
                elif total_pages > 100 and general_score > insurance_score and insurance_score < 3:
                    print(f"🔍 Large document with general dominance ({general_score} > {insurance_score})")
                    is_general_document = True
                    reason = "large_mixed_content"
                
                # NEW Check 6: Strong insurance indicators override everything else
                elif insurance_score >= 5:
                    print(f"🏥 Strong insurance indicators ({insurance_score}) - definitely insurance document")
                    is_general_document = False
                
                if is_general_document:
                    print(f"🎯 GENERAL DOCUMENT DETECTED - Routing to Direct LLM ({reason})")
                    return "direct_llm", {
                        "size_mb": size_mb, 
                        "reason": reason,
                        "pages": total_pages,
                        "insurance_score": insurance_score,
                        "general_score": general_score
                    }
                else:
                    print(f"🏥 Insurance-related content detected (Insurance={insurance_score} > General={general_score})")
                    
        except Exception as e:
            print(f"⚠️ Content analysis failed: {e}")
    
    # Strategy 3: Standard RAG for policy documents
    print(f"🏥 Insurance document detected - Using RAG pipeline")
    return "rag", {"size_mb": size_mb, "reason": "insurance_document"}