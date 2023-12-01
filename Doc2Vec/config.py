from supabase import create_client

### Input sources
ai_book_path = 'D:\ML2\RAG\data\AI-book.pdf'
base_url= "https://www.geeksforgeeks.org"
web_url = 'https://www.geeksforgeeks.org/machine-learning/'

### SUPABASE DB
supabase_url = "https://mufqacshyjgmzivznuwo.supabase.co"
supabase_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im11ZnFhY3NoeWpnbXppdnpudXdvIiwicm9sZSI6ImFub24iLCJpYXQiOjE2OTg0MzM2MTQsImV4cCI6MjAxNDAwOTYxNH0.iYqEsEQJWcQUu7KjYWBzrRKdZn23Vp_xlhkHomc85Tc"
client = create_client(supabase_url, supabase_key)
table_name = "combined_test"

headers = {
    "apikey": supabase_key,
    "Content-Type": "application/json"
}

endpoint = f"{supabase_url}/rest/v1/combined_test"

# Text splitter
chunk_size = 1000
chunk_overlap = 50