import requests
# import pypdf
import io
import fitz
from langchain.text_splitter import RecursiveCharacterTextSplitter

def get_and_chunk_pdf(url: str) -> list[str]:
    print(f"Downloading document from {url}...")
    try:
        #  Use stream=True to avoid loading the whole file into memory at once
        with requests.get(url, stream=True, timeout=30) as r:
            r.raise_for_status()

            # create an in-memory bytes buffer
            pdf_file_bytes = io.BytesIO()

            # Download the file in chunks and write to the buffer
            for chunk in r.iter_content(chunk_size=8192):
                pdf_file_bytes.write(chunk)

            # reset the buffer's position to the beginning for reading
            pdf_file_bytes.seek(0)
 
    except requests.exceptions.RequestException as e:
        print(f"Error downloading PDF: {e}")
        raise

    print("Successfully streamed PDF. Extracting text with PyMuPDF...")
    
    text=""
    # Use fitz to open the document from the in-memory bytes stream
    with fitz.open(stream=pdf_file_bytes, filetype="pdf") as doc:
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


    