# In app/vector_store.py

import asyncio
import hashlib
import chromadb # Import ChromaDB
from sentence_transformers import SentenceTransformer
# import google.generativeai as genai
from . import config
import os

# --- Globals ---
# We no longer need the Pinecone client
# 1. Initialize the ChromaDB client. This is lightweight.
#    Using an in-memory-only instance is fastest for a hackathon.
chroma_client = chromadb.Client()

print(f"Loading local embedding model: '{config.EMBEDDING_MODEL}'...")
embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
print("Local embedding model loaded successfully.")


# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
# EMBEDDING_MODEL_NAME = "models/text-embedding-004"
# EMBEDDING_DIMENSION = 768

def create_or_get_index(collection_name: str):
    """
    Gets or creates a ChromaDB collection. This is virtually instantaneous.
    """
    print(f"Accessing ChromaDB collection: '{collection_name}'")
    # 2. This one line replaces the entire slow create_index loop
    return chroma_client.get_or_create_collection(name=collection_name)

def embed_and_store(chunks: list[str], collection: chromadb.Collection):
    """
    Generates embeddings locally and stores them in a ChromaDB collection.
    """
    print(f"Generating embeddings for {len(chunks)} chunks locally...")

    # 4. Use the loaded model to create embeddings.
    #   This runs on your CPU. 'show_progress_bar' is very helpful for seeing progress.
    embeddings = embedding_model.encode(chunks, show_progress_bar=True)
    print(f"Storing {len(chunks)} embeddings into ChromaDB...")
    collection.add(
        # The model outputs a NumPy array, so we convert it to a list for ChromaDB.
        embeddings=embeddings.tolist(),
        documents=chunks,
        ids=[f"chunk_{i}" for i in range(len(chunks))]
    )
    print("Storage complete.")
    
    
    

def retrieve_context(query: str, collection: chromadb.Collection) -> str:
    """
    Generates an embedding for the query and retrieves context from ChromaDB.
    """
    print(f"Retrieving context for query: '{query}'")

    # 5. Embed the single query string locally.
    query_embedding = embedding_model.encode(query)

    # 6. Query the ChromaDB collection with the new embedding.
    results = collection.query(
        query_embeddings=[query_embedding.tolist()], # ChromaDB expects a list of embeddings
        n_results=config.TOP_K_RESULTS
    )

    # The result structure is slightly different from Pinecone but the idea is the same.
    # The relevant text is in the 'documents' key of the results dictionary.
    context = " ".join(results['documents'][0])
    return context


# The async wrapper function remains the same conceptually
async def retrieve_context_async(query: str, collection: chromadb.Collection) -> str:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None, retrieve_context, query, collection
    )