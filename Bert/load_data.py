### Please install these dependancies

# !pip install sentence-transformers
# !pip install openai==0.28
# !pip install pdfminer.six
# !pip install httpx
# !pip install supabase

import PyPDF2
import nltk
import config
import os
import httpx
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
from supabase import create_client, Client
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer

# Load the pretrained Sentence Transformers model
model = SentenceTransformer(config.model_name)


def list_pdf_files(pdf_path):
    """Lists all PDF files in the given directory."""
    pdf_files = [file for file in os.listdir(pdf_path) if file.endswith('.pdf')]
    return pdf_files


def preprocess_sentence(sentence):
    """Tokenizes and preprocesses a sentence."""
    tokens = word_tokenize(sentence.lower())
    return tokens


def chunk_text(text, chunk_size=1200, overlap=100):
    """Chunks text into sizes of about provided characters, ending at sentence boundaries."""
    chunks = []
    sentences = nltk.sent_tokenize(text)
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) > chunk_size and len(current_chunk) > 0:
            chunks.append(current_chunk)
            current_chunk = sentence
        else:
            current_chunk += (" " if current_chunk else "") + sentence

        if len(current_chunk) > chunk_size and len(chunks) > 0:
            last_chunk = chunks[-1]
            overlap_text = last_chunk[-overlap:]
            current_chunk = overlap_text + current_chunk

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def extract_text_by_page(pdf_path, dir_path):
    """Extracts text from PDF pages using PDFMiner and generates embeddings."""
    file_path = dir_path + pdf_path
    print(file_path)
    for page_number, page_layout in enumerate(extract_pages(file_path)):
        text = ""
        for element in page_layout:
            if isinstance(element, LTTextContainer):
                text += element.get_text()

        text_chunks = chunk_text(text)  # Assuming you have a function for chunking text
        for chunk in text_chunks:
            vector = model.encode(chunk).tolist()  # Encoding the chunk using your model
            yield chunk, vector, page_number


# Supabase setup
headers = {"apikey": config.SUPABASE_SERVICE_API_KEY, "Content-Type": "application/json"}

pdf_files = list_pdf_files(config.dir_path)

# Process each PDF and send data to Supabase
for pdf_path in pdf_files:
    for chunk, vector, page_number in extract_text_by_page(pdf_path, config.dir_path):
        # print(len(vector))
        data_to_insert = {
            "page_number": str(page_number),
            "vector_data": vector,
            "text": chunk,
            "text_book": pdf_path
        }
        endpoint = f"{config.SUPABASE_URL}/rest/v1/{config.supabase_table_name}"
        response = httpx.post(endpoint, headers=headers, json=data_to_insert)
        print(f"Stored data for page {page_number} in '{pdf_path}': {response.status_code}, {response.text}")
