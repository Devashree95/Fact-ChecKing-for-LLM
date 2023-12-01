-- Create table
CREATE TABLE books_pretrained_embeddings (
  source TEXT,
  vector_data vector(768),  -- Array of 768 floats for vector data
  text TEXT,
  text_book TEXT,
  page_number TEXT
);

-- Match function
CREATE OR REPLACE FUNCTION match_books_pretrained_embeddings(
  query_vector VECTOR,
  threshold FLOAT,
  match_count INT
)
RETURNS TABLE (
  source TEXT,
  text TEXT,
  text_book TEXT,
  page_number TEXT,
  cosine_similarity_score FLOAT
)
LANGUAGE sql STABLE
AS $$
  SELECT
    books_pretrained_embeddings.source,
    books_pretrained_embeddings.text,
    books_pretrained_embeddings.text_book,
    books_pretrained_embeddings.page_number,
    1 - (books_pretrained_embeddings.vector_data <=> query_vector) AS cosine_similarity_score
  FROM books_pretrained_embeddings
  WHERE 1 - (books_pretrained_embeddings.vector_data <=> query_vector) > threshold
  ORDER BY cosine_similarity_score DESC
  LIMIT match_count;
$$;