# Required imports for the script
import httpx
from sentence_transformers import SentenceTransformer
import re
import json
import config
import openai

# Set the OpenAI API key from the configuration
openai_api_key = config.openai_key

"""
    Queries the Supabase database using an embedding generated from the query document.
"""
def query_supabase_with_embedding(query_document, model):
    # Generate a vector for the query document
    query_vector = model.encode(query_document).tolist()  # Convert the text to an embedding list

    # Supabase setup
    SUPABASE_URL = config.SUPABASE_URL
    SUPABASE_SERVICE_API_KEY = config.SUPABASE_SERVICE_API_KEY

    headers = {
        "apikey": SUPABASE_SERVICE_API_KEY,
        "Content-Type": "application/json"
    }

    # Prepare the data payload
    data = {
        "query_vector": query_vector,
        "threshold": 0.25,
        "match_count": 2
    }

    # RPC endpoint for executing the function
    endpoint = f"{SUPABASE_URL}/rest/v1/rpc/{config.match_function_name}"

    # Execute the function via POST request
    response = httpx.post(endpoint, headers=headers, json=data)
    if response.status_code == 200:
        results = response.json()
        return [str(r) for r in results]
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return ["error in RAG"]


# Initialize the SentenceTransformer model
model = SentenceTransformer(config.model_name)


"""
    Extracts JSON or SQL code snippets from the provided text.
"""
def fetch_json_subparts(text):
    # This regex captures content between the outermost curly braces
    match = re.search(r'({.*})', text, re.DOTALL)
    if match:
        try:
            json_obj = json.loads(match.group(1))
            return json_obj
        except json.JSONDecodeError as e:
            print(f"ERROR in JSON ! {e}")
            return "error"
    # If not JSON, try to get SQL code
    markdown_match = re.search(r'```json(.*?)```', text, re.DOTALL)
    try:
        json_obj = json.loads(markdown_match.group(1))
        return json_obj
    except json.JSONDecodeError as e:
        print(f"ERROR in JSON ! {e}")
        return "error"

"""
    Continues a conversation with GPT-4 using the provided prompt.
"""
def continue_conversation(prompt):
    response = openai.ChatCompletion.create(model="gpt-4-0314", top_p=0,
                                            messages=prompt,
                                            temperature=0)
    return response['choices'][0]['message']['content'].strip()


"""
    Check the factual accuracy of a query using OpenAI's GPT-4 with context from retrieved documents.

    :param query: The query or statement to be checked.
    :param documents: A list of documents that provide context for the query.
    :param openai_api_key: Your OpenAI API key.
    :return: The model's response regarding the factual accuracy of the query.
"""
def check_fact_with_context(query, documents, openai_api_key):
    openai.api_key = openai_api_key

    # Prepare the context by concatenating document contents
    context = ' /n '.join(documents)  # You might need to adjust this based on how your documents are structured

    # Formulate the prompt for GPT-4
    prompt = f"Based on the following information: {context}\nCan you tell me if this statement is true or false? '{query}'"
    prompt = f""" # Your Job is to work as Fact checkers and carefuly understand given Source Text as ground truth information and query as claim or text needs to validate.
                ## You should not use information outside of given Source Text and so step by step analysis in your mind and **strictly** provide output in following json format:
                     ## Carefully evaluate info even for minute details
                     ## Rule: You are not allowed to used accurate or inaccurate in classification if you dont knopw the reference where source fact lies
                     f"##Query / info to validate : '{query}'. " \
                     f"## Source Text: '{context}'. " \

                    # Expected JSON output format:
                     ```json
                        {{
                        classfication : "label" -  3 options for lables "accurate", "inaccurate" and "information not found in source" This can be classified only after reason
                        reason : "any supported reasoning based Source Text else say N/A"
                        reference: " each document from Source text will have text_book and page_number info so provide that in string. if its label 3 info not found then provide N/A "

                        }}
                     ```

                     """

    history = [
        {"role": "system", "content": "You are Fact check evaluator!"},
        {"role": "user", "content": prompt}]

    response = continue_conversation(history)
    return response


## Test the fact-checking process with examples
test_examples = [
    [
        "Deductive reasoning is a form of illogical thinking that uses unrelated observations to arrive at a specific conclusion. This type of reasoning is common in descriptive science.",
        "inaccurate"]
]

for i, example in enumerate(test_examples):
    query = example[0]
    documents = query_supabase_with_embedding(query, model)

    result = check_fact_with_context(query, documents, openai_api_key)
    print("* " * 40)
    print("example No :", i + 1)
    print(query)
    print(fetch_json_subparts(result))
    print("expected output : ", example[1])
