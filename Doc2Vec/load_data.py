import config
import create_embeddings
from gensim.models.doc2vec import Doc2Vec
import data_import

pdf_path = config.ai_book_path
embedding = create_embeddings.embedding
# Load the model
model = Doc2Vec.load("jesc102_model.d2v")

# Run to store web URL data to supabase
data_import.dataImport.importUrl()

# Run to store PDF data to supabase
data_import.dataImport.importPdf(pdf_path)

