from supabase import create_client

### Input sources
ai_book_path = '/data'
base_url= "https://www.geeksforgeeks.org"
web_url = 'https://www.geeksforgeeks.org/machine-learning/'

### SUPABASE DB

#Please provide you own URL and key here while running
supabase_url = "https://mufqacshyjgmzivznuwo.supabase.co"
supabase_key = "Please provide your key here"
client = create_client(supabase_url, supabase_key)
#Please change the table name to your table name
table_name = "combined_test"

headers = {
    "apikey": supabase_key,
    "Content-Type": "application/json"
}

# Please change the table name to your table name
endpoint = f"{supabase_url}/rest/v1/combined_test"

### Config for text splitter
# Text splitter
chunk_size = 1000
chunk_overlap = 50

### Open AI
openai_key = "Please provide open AI key here"
