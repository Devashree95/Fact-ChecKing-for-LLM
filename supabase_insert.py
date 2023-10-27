# pip install supabase

from supabase import create_client

# Initialize the Supabase client
supabase_url = "https://mufqacshyjgmzivznuwo.supabase.co"
supabase_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im11ZnFhY3NoeWpnbXppdnpudXdvIiwicm9sZSI6ImFub24iLCJpYXQiOjE2OTg0MzM2MTQsImV4cCI6MjAxNDAwOTYxNH0.iYqEsEQJWcQUu7KjYWBzrRKdZn23Vp_xlhkHomc85Tc"
client = create_client(supabase_url, supabase_key)

# Define your vector data (for example, as a Python dictionary)
vector_data = {
    "x": 1.0,
    "y": 2.0,
    "z": 3.0,
}

# Insert the data into the Supabase table
table_name = "test_table"  # Replace with the actual table name

response = client.from_(table_name).upsert([vector_data]).execute()
print(response)
