### Please install these dependancies

# !pip install sentence-transformers
# !pip install openai==0.28
# !pip install pdfminer.six
# !pip install httpx
# !pip install supabase

# Make necessary imports
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

# function to list files in a given directory
def list_pdf_files(pdf_path):
    """Lists all PDF files in the given directory."""
    pdf_files = [file for file in os.listdir(pdf_path) if file.endswith('.pdf')]
    return pdf_files

# Tokenize and prepocess the sentence
def preprocess_sentence(sentence):
    """Tokenizes and preprocesses a sentence."""
    tokens = word_tokenize(sentence.lower())
    return tokens

"""Chunks text into sizes of about provided characters, ending at sentence boundaries.
    Parameters:
    text (str): The text to be chunked.
    chunk_size (int): Target size for each chunk in characters.
    overlap (int): Number of characters from the end of one chunk to overlap with the beginning of the next chunk.

    Returns:
    list: A list of text chunks.
    """
def chunk_text(text, chunk_size=1200, overlap=100):
    # Initialize an empty list to store the chunks
    chunks = []
    sentences = nltk.sent_tokenize(text)
    current_chunk = ""

    # Iterate through each sentence in the tokenized text.
    for sentence in sentences:
        # If adding a sentence to the current chunk exceeds the chunk size
        # and the current chunk is not empty, add the current chunk to the chunks list.
        # Then, start a new chunk with the current sentence.
        if len(current_chunk) + len(sentence) > chunk_size and len(current_chunk) > 0:
            chunks.append(current_chunk)
            current_chunk = sentence
        else:
            # If the current chunk plus the new sentence is within the limit,
            # add the sentence to the current chunk.
            current_chunk += (" " if current_chunk else "") + sentence

        # If the length of the current chunk exceeds the chunk size and there are already chunks in the list,
        # take the last part of the previous chunk (as defined by the overlap) and add it to the current chunk.
        # This creates an overlap between the end of the previous chunk and the start of the current chunk.
        if len(current_chunk) > chunk_size and len(chunks) > 0:
            last_chunk = chunks[-1]
            overlap_text = last_chunk[-overlap:]
            current_chunk = overlap_text + current_chunk

    # After processing all sentences, if there is any remaining text in the current chunk, add it to the chunks list.
    if current_chunk:
        chunks.append(current_chunk)

    return chunks


    """
    Extracts text from each page of a PDF file and generates embeddings for the extracted text.

    Parameters:
    pdf_path (str): The path to the PDF file relative to the directory path.
    dir_path (str): The directory path where the PDF file is located.

    Yields:
    tuple: A tuple containing the text chunk, its embedding, and the page number.
    """
def extract_text_by_page(pdf_path, dir_path):
    file_path = dir_path + pdf_path
    # Iterate through each page in the PDF file.
    for page_number, page_layout in enumerate(extract_pages(file_path)):
        text = ""
        # Iterate through each element in the page layout
        for element in page_layout:
            if isinstance(element, LTTextContainer):
                # Append the text of the element to the overall text of the page.
                text += element.get_text()

        # Chunk the extracted text into smaller pieces.
        text_chunks = chunk_text(text) 

        # Iterate through each chunk and generate its embedding.
        for chunk in text_chunks:
            # Generate the embedding for the chunk of text and convert it to a list.
            vector = model.encode(chunk).tolist()
            
            # Yield a tuple containing the text chunk, its embedding, and the page number.
            yield chunk, vector, page_number


# Supabase setup
headers = {"apikey": config.SUPABASE_SERVICE_API_KEY, "Content-Type": "application/json"}

pdf_files = list_pdf_files(config.dir_path)

# Process each PDF and send data to Supabase
# Iterate through each PDF file in the list.
for pdf_path in pdf_files:
    for chunk, vector, page_number in extract_text_by_page(pdf_path, config.dir_path):

        # Prepare the data to be inserted into the database.
        data_to_insert = {
            "page_number": str(page_number),
            "vector_data": vector,
            "text": chunk,
            "text_book": pdf_path
        }

        # Construct the endpoint URL for posting data to Supabase.
        endpoint = f"{config.SUPABASE_URL}/rest/v1/{config.supabase_table_name}"

        # Use the httpx library to send a POST request to the Supabase endpoint.
        response = httpx.post(endpoint, headers=headers, json=data_to_insert)
        print(f"Stored data for page {page_number} in '{pdf_path}': {response.status_code}, {response.text}")
