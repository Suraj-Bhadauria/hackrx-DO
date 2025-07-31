import asyncio
import hashlib
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from . import config
import os
import concurrent.futures
import time
import torch

# --- Globals with GPU detection ---
print("🔍 Checking for GPU availability...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🖥️ Using device: {device.upper()}")

chroma_client = chromadb.Client()

print("Loading Bi-Encoder embedding model...")
embedding_model = SentenceTransformer(config.EMBEDDING_MODEL, device=device)
print(f"✅ Bi-Encoder loaded on {device.upper()}")

print("Loading Cross-Encoder model...")
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L6-v2', device=device)
print(f"✅ Cross-Encoder loaded on {device.upper()}")

def create_or_get_index(collection_name: str):
    """Gets or creates a ChromaDB collection."""
    print(f"Accessing ChromaDB collection: '{collection_name}'")
    return chroma_client.get_or_create_collection(name=collection_name)

def embed_and_store(documents: list[str], metadatas: list[dict], ids: list[str], collection: chromadb.Collection):
    """
    GPU-accelerated embedding generation with detailed timing
    """
    total_chunks = len(documents)
    start_time = time.time()
    print(f"🚀 GPU-accelerated embedding generation for {total_chunks} chunks...")
    print(f"⏰ Start time: {time.strftime('%H:%M:%S', time.localtime(start_time))}")
    
    # Larger batch size for GPU
    BATCH_SIZE = 200 if device == "cuda" else 100
    batch_size_internal = 64 if device == "cuda" else 32
    
    total_batches = (total_chunks - 1) // BATCH_SIZE + 1
    
    for i in range(0, total_chunks, BATCH_SIZE):
        batch_start_time = time.time()
        batch_num = i // BATCH_SIZE + 1
        
        batch_end = min(i + BATCH_SIZE, total_chunks)
        batch_docs = documents[i:batch_end]
        batch_meta = metadatas[i:batch_end]
        batch_ids = ids[i:batch_end]
        
        print(f"📊 Processing batch {batch_num}/{total_batches} ({len(batch_docs)} chunks)")
        
        # GPU-accelerated embedding generation
        batch_embeddings = embedding_model.encode(
            batch_docs, 
            show_progress_bar=False,
            batch_size=batch_size_internal,
            convert_to_numpy=True,
            device=device
        )
        
        # Store this batch
        collection.add(
            embeddings=batch_embeddings.tolist(),
            documents=batch_docs,
            metadatas=batch_meta,
            ids=batch_ids
        )
        
        batch_time = time.time() - batch_start_time
        chunks_per_second = len(batch_docs) / batch_time
        print(f"✅ Batch {batch_num} completed in {batch_time:.2f}s ({chunks_per_second:.1f} chunks/sec)")
    
    total_time = time.time() - start_time
    avg_chunks_per_second = total_chunks / total_time
    print(f"🎉 All {total_chunks} embeddings completed!")
    print(f"⏱️ Total time: {total_time:.2f} seconds")
    print(f"📈 Average speed: {avg_chunks_per_second:.1f} chunks/second")
    print(f"🏁 End time: {time.strftime('%H:%M:%S', time.localtime())}")

async def embed_and_store_async(documents: list[str], metadatas: list[dict], ids: list[str], collection: chromadb.Collection):
    """
    Ultra-fast async GPU embedding generation with parallel processing
    """
    total_chunks = len(documents)
    start_time = time.time()
    print(f"⚡ TURBO MODE: Async GPU embedding generation for {total_chunks} chunks...")
    print(f"⏰ Start time: {time.strftime('%H:%M:%S', time.localtime(start_time))}")
    
    def generate_embeddings_batch(batch_docs):
        batch_start = time.time()
        # Use larger batch sizes for GPU
        internal_batch_size = 128 if device == "cuda" else 64
        embeddings = embedding_model.encode(
            batch_docs, 
            show_progress_bar=False, 
            batch_size=internal_batch_size,
            device=device,
            convert_to_numpy=True
        )
        batch_time = time.time() - batch_start
        chunks_per_sec = len(batch_docs) / batch_time if batch_time > 0 else 0
        print(f"🔥 Batch of {len(batch_docs)} chunks: {batch_time:.2f}s ({chunks_per_sec:.1f} chunks/sec)")
        return embeddings
    
    # Larger batches for GPU, more workers
    BATCH_SIZE = 300 if device == "cuda" else 150
    max_workers = 4 if device == "cuda" else 3
    
    batches = [documents[i:i+BATCH_SIZE] for i in range(0, len(documents), BATCH_SIZE)]
    print(f"🎯 Processing {len(batches)} batches with {max_workers} parallel workers")
    
    # Process batches in parallel
    loop = asyncio.get_running_loop()
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        embedding_start = time.time()
        tasks = [
            loop.run_in_executor(executor, generate_embeddings_batch, batch)
            for batch in batches
        ]
        
        embeddings_batches = await asyncio.gather(*tasks)
        embedding_time = time.time() - embedding_start
        print(f"🚀 Parallel embedding generation: {embedding_time:.2f}s")
    
    # Combine and store
    store_start = time.time()
    all_embeddings = []
    for batch_embeddings in embeddings_batches:
        all_embeddings.extend(batch_embeddings.tolist())
    
    collection.add(
        embeddings=all_embeddings,
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    store_time = time.time() - store_start
    
    total_time = time.time() - start_time
    avg_chunks_per_second = total_chunks / total_time
    
    print(f"💾 Storage time: {store_time:.2f}s")
    print(f"🎉 TURBO COMPLETE: {total_chunks} embeddings in {total_time:.2f}s!")
    print(f"🏆 FINAL SPEED: {avg_chunks_per_second:.1f} chunks/second")
    print(f"🏁 End time: {time.strftime('%H:%M:%S', time.localtime())}")

# Enhanced retrieval function remains the same...
def retrieve_context(query: str, collection: chromadb.Collection) -> str:
    """Enhanced retrieval with timing"""
    start_time = time.time()
    print(f"🔍 Retrieving context for: '{query[:50]}...'")
    
    query_lower = query.lower()
    
    is_yes_no = any(query.lower().startswith(word) for word in ['is', 'are', 'does', 'do', 'can', 'will', 'would', 'should'])
    is_numerical = any(word in query_lower for word in ['how much', 'how many', 'what is the cost', 'price', 'amount', 'limit'])
    is_definition = any(word in query_lower for word in ['what is', 'what are', 'define', 'meaning'])
    
    expanded_terms = []
    if 'cost' in query_lower or 'price' in query_lower or 'fee' in query_lower:
        expanded_terms.extend(['cost', 'price', 'amount', 'fee', 'charge'])
    if 'time' in query_lower or 'period' in query_lower or 'duration' in query_lower:
        expanded_terms.extend(['days', 'weeks', 'months', 'period', 'time', 'duration'])
    if 'cover' in query_lower or 'include' in query_lower:
        expanded_terms.extend(['coverage', 'include', 'cover', 'benefits'])
    
    if expanded_terms:
        enhanced_query = f"{query} {' '.join(expanded_terms[:3])}"
    else:
        enhanced_query = query
    
    if is_definition or len(query.split()) > 10:
        k_candidates = 12
        k_final = 6
    elif is_numerical:
        k_candidates = 10
        k_final = 5
    else:
        k_candidates = 8
        k_final = 4
    
    query_with_prefix = f"Represent this sentence for searching relevant passages: {enhanced_query}"
    query_embedding = embedding_model.encode(query_with_prefix)
    
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=k_candidates
    )
    
    initial_documents = results['documents'][0]
    if not initial_documents:
        return ""
    
    cross_inp = [[query, doc] for doc in initial_documents]
    cross_scores = cross_encoder.predict(cross_inp)
    
    doc_scores = list(zip(initial_documents, cross_scores))
    doc_scores.sort(key=lambda x: x[1], reverse=True)
    
    final_docs = [doc for doc, score in doc_scores[:k_final]]
    
    context_parts = []
    for i, doc in enumerate(final_docs):
        context_parts.append(f"[Section {i+1}] {doc.strip()}")
    
    final_context = "\n\n".join(context_parts)
    
    retrieval_time = time.time() - start_time
    print(f"⚡ Context retrieved in {retrieval_time:.2f}s ({len(final_docs)} sections)")
    
    return final_context

# The async wrapper function remains the same conceptually
async def retrieve_context_async(query: str, collection: chromadb.Collection) -> str:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None, retrieve_context, query, collection
    )