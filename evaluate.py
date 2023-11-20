import config
import create_embeddings
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import data_import

pdf_path = config.ai_book_path
embedding = create_embeddings.embedding
# Load the model
model = Doc2Vec.load("jesc102_model.d2v")

new_sentence = "Hello."
vector = model.infer_vector(embedding.preprocess_sentence(new_sentence))
print(vector)

# data_import.dataImport.importUrl()

data_import.dataImport.importPdf(pdf_path)

