import requests
# import pypdf
import io
import fitz
from langchain.text_splitter import RecursiveCharacterTextSplitter

def get_and_chunk_pdf(url: str) -> list[str]:
    print(f"Downloading document from {url}...")
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()  
    except requests.exceptions.RequestException as e:
        print(f"Error downloading PDF: {e}")
        raise

    print("Successfully downloaded PDF. Extracting text with PyMuPDF...")
    
    # Read PDF content from the in-memory response content
    text=""
    pdf_file = io.BytesIO(response.content)

    # Use fitz to open the document
    with fitz.open(stream=pdf_file, filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()

    if not text:
        print("Error: No text could be extracted from the PDF.")
        return []
    
    print(f"Successfully extracted {len(text)} characters from the document.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )

    chunks = text_splitter.split_text(text)
    print(f"Split text into {len(chunks)} chunks.")
    return chunks


    