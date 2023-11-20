import config
import create_embeddings
import PyPDF2
import requests
from bs4 import BeautifulSoup
#import storeToDB
import requests
from bs4 import BeautifulSoup
import httpx
import json
import preprocess
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

pdf_path = config.ai_book_path
url = config.web_url
embedding = create_embeddings.embedding
# store= storeToDB.saveToDb
chunking = preprocess.preprocess
model = Doc2Vec.load("jesc102_model.d2v")

class dataImport:

    def importPdf(pdf):
        tagged_data = list(create_embeddings.embedding.extract_text_by_page(pdf))
        model = Doc2Vec.load("jesc102_model.d2v")
        pages_text = list(tagged_data)
        for i in range(len(tagged_data)):
            vector = model.dv[i].tolist()  # Convert to list assuming it's a numpy array
            tag = i

            # Convert your vector to a JSON string.
            data_to_insert = {
                "source": tag,
                "text": pages_text[i],
                "vector": json.dumps(vector)
            }

            response = httpx.post(config.endpoint, headers=config.headers, json=data_to_insert)
            print(f"Stored vector for '{tag}': {response}")
        return tagged_data

    def importUrl():
        response = requests.get(url)
        sections = []

        if response.status_code == 200:
            # Parse the HTML content of the page
            soup = BeautifulSoup(response.content, 'html.parser')

            # Find the <ul> tag with the class 'leftBarList'
            left_bar_list = soup.find('ul', class_='leftBarList')

            # Find all <a> tags within the left bar list
            links = left_bar_list.find_all('a') if left_bar_list else []
            for link in links:
                # Ensure that only full URLs are printed
                href = link.get('href')
                full_url = href if href.startswith('http') else config.base_url + f"{href}"

                page_response = requests.get(full_url)
                if page_response.status_code == 200:
                    # Parse the HTML content of the page
                    page_soup = BeautifulSoup(page_response.content, 'html.parser')

                    # Find all <p> tags on the page and get their text content
                    paragraphs = page_soup.find_all('p')
                    paragraph_texts = [p.get_text(separator='\n', strip=True) for p in paragraphs]

                    # Combine all paragraph texts into a single string
                    full_text = '\n\n'.join(paragraph_texts)
                    section = {"text": full_text, "source": full_url}
                    sections.append({"text": full_text, "source": full_url})
                    chunks_ds = chunking.chunk_section(section, chunk_size=config.chunk_size,
                                                       chunk_overlap=config.chunk_overlap)
                    for chunk in chunks_ds:
                        vector = model.infer_vector(embedding.preprocess_sentence(chunk["text"])).tolist()
                        vector_data = json.dumps(vector)
                        # print(vector_data)

                        data_to_insert = {
                            "source": chunk["source"],
                            "text": chunk["text"],
                            "vector": vector_data
                        }

                        # response = client.from_(table_name).upsert([vector_data]).execute()

                        response = httpx.post(config.endpoint, headers=config.headers, json=data_to_insert)
                else:
                    print(f"Failed to retrieve the page from {full_url}. Status code: {page_response.status_code}")

        else:
            print(f"Failed to retrieve the webpage. Status code: {response.status_code}")






