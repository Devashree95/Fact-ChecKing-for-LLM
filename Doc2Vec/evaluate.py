# make necessary imports
import requests
from bs4 import BeautifulSoup
import httpx
import json
import time
import re
import matplotlib.pyplot as plt
from langchain.text_splitter import RecursiveCharacterTextSplitter
from supabase import create_client
import PyPDF2
import nltk
from nltk.tokenize import word_tokenize
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.corpus import stopwords
import os
import openai
import json
from supabase import create_client, Client
from IPython.display import display
from sklearn.metrics.pairwise import cosine_similarity

embedding = create_embeddings.embedding
openai.api_key = config.openai_key


# # A decorator function to measure and print the execution time of functions.
def time_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"{func.__name__} took {elapsed_time:.6f} seconds to execute.")
        return result

    return wrapper

# Function to continue a conversation using OpenAI's GPT-4.
@time_decorator
def continue_conversation(prompt):
    response = openai.ChatCompletion.create(model="gpt-4", top_p=0,
                                            messages=prompt,
                                            temperature=0)
    return response['choices'][0]['message']['content'].strip()


# Supabase details
url: str = config.supabase_url
key: str = config.supabase_key

# Create a client to connect to Supabase
supabase: Client = create_client(url, key)

# Data retrieval from Supabase
ids = []
sources = []
vectors = []

response = supabase.table("vector_final").select("id, source, vector").execute()
if response.data:
    for row in response.data:
        # Extracting each column's value and appending to the respective lists
        ids.append(row['id'])
        sources.append(row['source'])
        vectors.append(row['vector'])

# Prepare the vectors for cosine similarity
vectors = [ast.literal_eval(vector) for vector in vectors]
vectors = np.array(vectors)

# Load the pre-trained Doc2Vec model
model = Doc2Vec.load("jesc102_model.d2v")

# Load ground truth and test data from Excel files
truth = pd.read_excel('/test/llm_testing.xlsx')
test = pd.read_excel('/test/questions.xlsx')

# Main process for checking answers against a test dataset
answers = []
for q in test["Question"]:
    top5_data = []  # list to store top 5 contexts
    initial_user_prompt = q
    new_system_prompt = "Answer the question asked by the user"

    print(q)

    history = [
        {"role": "system", "content": new_system_prompt},
        {"role": "user", "content": initial_user_prompt}]

    # Get the answer of question from LLM
    response_by_LLM = continue_conversation(history)

    # Convert the response to vector embeddings
    vector_sentence = model.infer_vector(embedding.preprocess_sentence(response_by_LLM))
    sr = []

    vector_sentence = np.array(vector_sentence)
    vector_sentence_2d = vector_sentence.reshape(1, -1) # reshape the vector

    # calculate the cosine similarity
    similarities = cosine_similarity(vector_sentence_2d, vectors)

    #get index for top 5 contexts
    top_5_indexes = np.argsort(similarities[0])[-5:][::-1]
    for index in top_5_indexes:
        sr.append(sources[index])

    res = []
    for i in range(len(sr)):
        # fetch data for top 5 contexts from supabase
        data = supabase.table("vector_final").select("text").eq("source", sr[i]).execute()
        top5_data.append(data)

        # Extract the relevant text
        relevant_text_from_source = [row['text'] for row in top5_data[i].data]

        initial_user_prompt = q

        #Prompt given to GPT4
        new_system_prompt = f"Compare the following response with the provided source text using semantic check and determine its accuracy. " \
                            f"Response: '{response_by_LLM}'. " \
                            f"Source Text: '{relevant_text_from_source}'. " \
                            f"If the response accurately reflects the source text, reply 'accurate'. " \
                            f"If the response does not accurately reflect the source text , reply 'inaccurate'. " \
                            f"If the response's content is not found in the source text, reply 'information not found in source'." \
                            f"If the response is not relevant to question, reply 'response is irrelevant to question'. "

        history = [
            {"role": "system", "content": new_system_prompt},
            {"role": "user", "content": initial_user_prompt}]

        response = continue_conversation(history)
        res.append(response)
        print(res)
    # If accurate answer is found in any of top 5 contexts, tag the answer as accurate
    if 'Accurate' in res:
        answers.append("Final check: Accurate")
    else:
        answers.append("Final check: " + response)

## Claculate the accuracy
# Compare answers from LLM with expected answers and calculate the accuracy
acc = 1
for i in range(len(answers)):
    print(truth['Answer'][i])
    print(answers[i])
    if (truth['Answer'][i] == answers[i]):
        acc += 1
print("Accuracy : ", acc / len(answers))
