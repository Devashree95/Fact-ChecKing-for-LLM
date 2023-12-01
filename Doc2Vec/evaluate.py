# make necessary imports
import openai
from gensim.models.doc2vec import Doc2Vec
import create_embeddings
from supabase import create_client, Client
from sklearn.metrics.pairwise import cosine_similarity
import config
import time
import ast
import numpy as np
import pandas as pd

embedding = create_embeddings.embedding
openai.api_key = config.openai_key


# Define a function to continue the conversation
def time_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"{func.__name__} took {elapsed_time:.6f} seconds to execute.")
        return result

    return wrapper


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

# Now, ids, sources, and vectors have the data from their respective columns.
vectors = [ast.literal_eval(vector) for vector in vectors]
vectors = np.array(vectors)

model = Doc2Vec.load("jesc102_model.d2v")

truth = pd.read_excel('/test/llm_testing.xlsx')
test = pd.read_excel('/test/questions.xlsx')

answers = []
for q in test["Question"]:
    top5_data = []
    initial_user_prompt = q
    new_system_prompt = "Answer the question asked by the user"

    print(q)

    history = [
        {"role": "system", "content": new_system_prompt},
        {"role": "user", "content": initial_user_prompt}]

    response_by_LLM = continue_conversation(history)
    vector_sentence = model.infer_vector(embedding.preprocess_sentence(response_by_LLM))
    sr = []

    vector_sentence = np.array(vector_sentence)
    vector_sentence_2d = vector_sentence.reshape(1, -1)

    similarities = cosine_similarity(vector_sentence_2d, vectors)

    # Find the index of the most similar paragraph
    most_similar_index = np.argmax(similarities)

    top_5_indexes = np.argsort(similarities[0])[-5:][::-1]
    for index in top_5_indexes:
        sr.append(sources[index])

    res = []
    for i in range(len(sr)):
        data = supabase.table("vector_final").select("text").eq("source", sr[i]).execute()
        top5_data.append(data)

        # Extract the relevant text
        relevant_text_from_source = [row['text'] for row in top5_data[i].data]

        initial_user_prompt = q

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
    if 'Accurate' in res:
        answers.append("Final check: Accurate")
    else:
        answers.append("Final check: " + response)

## Claculate the accuracy

acc = 1
for i in range(len(answers)):
    print(truth['Answer'][i])
    print(answers[i])
    if (truth['Answer'][i] == answers[i]):
        acc += 1
print("Accuracy : ", acc / len(answers))
