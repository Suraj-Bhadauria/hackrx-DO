import asyncio
import hashlib
import chromadb # Import ChromaDB
from sentence_transformers import SentenceTransformer, CrossEncoder
from . import config
import os

# --- Globals ---
chroma_client = chromadb.Client()

print("Loading Bi-Encoder embedding model (for initial search)...")
embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
print("Bi-Encoder loaded.")


print("Loading Cross-Encoder model (for re-ranking)...")
# ms-marco-MiniLM models are small, fast, and very effective for re-ranking
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L6-v2')
print("Cross-Encoder loaded.")

def create_or_get_index(collection_name: str):
    """
    Gets or creates a ChromaDB collection. This is virtually instantaneous.
    """
    print(f"Accessing ChromaDB collection: '{collection_name}'")
    return chroma_client.get_or_create_collection(name=collection_name)

# --- THIS IS THE MODIFIED FUNCTION ---
def embed_and_store(
    documents: list[str], 
    metadatas: list[dict], 
    ids: list[str], 
    collection: chromadb.Collection
):
    """
    Generates embeddings locally and stores them, along with metadata, in a ChromaDB collection.
    
    Args:
        documents: The list of text chunks to embed.
        metadatas: The list of corresponding metadata dictionaries for each chunk.
        ids: The list of unique IDs for each chunk.
        collection: The ChromaDB collection object.
    """
    print(f"Generating embeddings for {len(documents)} chunks locally...")

    # 1. Use the 'documents' list (which contains the text) to create embeddings.
    embeddings = embedding_model.encode(documents, show_progress_bar=True)
    
    print(f"Storing {len(documents)} embeddings into ChromaDB...")
    
    # 2. Call collection.add with all the correctly formatted data.
    collection.add(
        embeddings=embeddings.tolist(), # Embeddings must be a list
        documents=documents,           # The text chunks
        metadatas=metadatas,           # The crucial metadata
        ids=ids                        # The pre-generated unique IDs
    )
    print("Storage complete.")


def retrieve_context(query: str, collection: chromadb.Collection) -> str:
    """
    Enhanced retrieval with query analysis and adaptive K selection.
    """
    print(f"Retrieving context for query: '{query}'")
    
    # Simple query analysis - no external function needed
    query_lower = query.lower()
    
    # Determine query type and adjust strategy
    is_yes_no = any(query.lower().startswith(word) for word in ['is', 'are', 'does', 'do', 'can', 'will', 'would', 'should'])
    is_numerical = any(word in query_lower for word in ['how much', 'how many', 'what is the cost', 'price', 'amount', 'limit'])
    is_definition = any(word in query_lower for word in ['what is', 'what are', 'define', 'meaning'])
    
    # Query expansion for better retrieval
    expanded_terms = []
    if 'cost' in query_lower or 'price' in query_lower or 'fee' in query_lower:
        expanded_terms.extend(['cost', 'price', 'amount', 'fee', 'charge'])
    if 'time' in query_lower or 'period' in query_lower or 'duration' in query_lower:
        expanded_terms.extend(['days', 'weeks', 'months', 'period', 'time', 'duration'])
    if 'cover' in query_lower or 'include' in query_lower:
        expanded_terms.extend(['coverage', 'include', 'cover', 'benefits'])
    
    # Create enhanced query
    if expanded_terms:
        enhanced_query = f"{query} {' '.join(expanded_terms[:3])}"  # Limit to avoid too long queries
    else:
        enhanced_query = query
    
    # Adaptive K based on query complexity and type
    if is_definition or len(query.split()) > 10:
        k_candidates = 12
        k_final = 6
    elif is_numerical:
        k_candidates = 10
        k_final = 5
    else:
        k_candidates = 8
        k_final = 4
    
    # BGE prefix for better embedding
    query_with_prefix = f"Represent this sentence for searching relevant passages: {enhanced_query}"
    query_embedding = embedding_model.encode(query_with_prefix)
    
    # Initial retrieval
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=k_candidates
    )
    
    initial_documents = results['documents'][0]
    if not initial_documents:
        return ""
    
    print(f"Retrieved {len(initial_documents)} candidates. Re-ranking...")
    
    # Re-ranking with Cross-Encoder
    cross_inp = [[query, doc] for doc in initial_documents]
    cross_scores = cross_encoder.predict(cross_inp)
    
    # Combine and sort by relevance
    doc_scores = list(zip(initial_documents, cross_scores))
    doc_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Select top results
    final_docs = [doc for doc, score in doc_scores[:k_final]]
    
    # Format context with markers for better LLM understanding
    context_parts = []
    for i, doc in enumerate(final_docs):
        # Add section markers to help LLM reference specific parts
        context_parts.append(f"[Section {i+1}] {doc.strip()}")
    
    final_context = "\n\n".join(context_parts)
    print(f"Final context assembled from {len(final_docs)} sections.")
    
    return final_context

# The async wrapper function remains the same conceptually
async def retrieve_context_async(query: str, collection: chromadb.Collection) -> str:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None, retrieve_context, query, collection
    )