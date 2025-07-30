import requests
import io
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from collections import defaultdict
from transformers import AutoTokenizer

# Initialize tokenizer once to be reused
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

def num_tokens(text: str) -> int:
    """Helper function to count tokens for the splitter."""
    return len(tokenizer.encode(text, add_special_tokens=False))

def get_and_chunk_pdf(url: str) -> list[dict]:
    """
    Downloads a PDF from a URL, cleans common headers/footers, and chunks it
    semantically page by page for optimal RAG performance.

    Args:
        url: The URL of the PDF document.

    Returns:
        A list of dictionary objects, where each dictionary represents a
        chunk with its text and metadata.
    """
    print(f"📥 Downloading document from {url}...")
    try:
        with requests.get(url, stream=True, timeout=30) as r:
            r.raise_for_status()
            # Use r.content for simplicity as we need to read it multiple times
            pdf_file_bytes = io.BytesIO(r.content)
    except requests.exceptions.RequestException as e:
        print(f"❌ Error downloading PDF: {e}")
        raise

    print("✅ Successfully downloaded PDF. Analyzing for headers/footers...")

    # --- Step 1: Identify and Filter Headers/Footers ---
    repeated_line_counts = defaultdict(int)
    total_pages = 0
    # First pass: Count line occurrences across all pages
    with fitz.open(stream=pdf_file_bytes, filetype="pdf") as doc:
        total_pages = len(doc)
        if total_pages == 0:
            print("⚠️ The PDF document is empty.")
            return []
            
        for page in doc:
            # Using page.get_text("text") is simpler and more robust
            page_text = page.get_text("text")
            # Create a set of unique, stripped lines for the page
            lines_on_page = {line.strip() for line in page_text.splitlines() if line.strip()}
            for line in lines_on_page:
                repeated_line_counts[line] += 1

    # A line is considered a header/footer if it appears on every single page
    repeated_lines = {
        line for line, count in repeated_line_counts.items()
        # A short line is less likely to be a meaningful header/footer
        if count == total_pages and len(line) > 10
    }

    # --- Step 2: Configure the Semantic Text Splitter ---
    text_splitter = RecursiveCharacterTextSplitter(
        # Define a hierarchy of separators to split on, from most to least important
        separators=["\n\n", "\n", ". ", " ", ""],
        # Target chunk size, optimized for models like all-MiniLM
        chunk_size=350,  # Smaller chunks for more precise retrieval
        # Maintain some overlap to preserve context between chunks
        chunk_overlap=75,  # Increased overlap to maintain context
        length_function=num_tokens,
        is_separator_regex=False,
    )

    # --- Step 3: Process Page by Page and Chunk Semantically ---
    all_chunks = []
    with fitz.open(stream=pdf_file_bytes, filetype="pdf") as doc:
        for page_num, page in enumerate(doc, start=1):
            # A. Get the full text of the page
            page_text = page.get_text("text")

            # B. Clean the page text by removing identified headers/footers
            lines = page_text.splitlines()
            clean_lines = [line for line in lines if line.strip() not in repeated_lines]
            clean_page_text = "\n".join(clean_lines).strip()

            # Skip pages that are empty after cleaning
            if not clean_page_text:
                continue

            # C. Split the cleaned text of the ENTIRE page at once
            # This is the key change: giving the splitter full context.
            subchunks = text_splitter.split_text(clean_page_text)

            for i, chunk_text in enumerate(subchunks):
                all_chunks.append({
                    "text": chunk_text,
                    "metadata": {
                        "source": url,
                        "page": page_num,
                        "chunk_index": i, # More descriptive key
                    }
                })

    if not all_chunks:
        print("⚠️ No usable chunks were created from the document.")
        return []

    print(f"✅ Successfully created {len(all_chunks)} semantic chunks with metadata.")
    return all_chunks