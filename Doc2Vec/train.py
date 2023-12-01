# Import necessary modules and classes.
from data_import import dataImport
import config
from gensim.models.doc2vec import Doc2Vec

pdf_path = config.ai_book_path

# Use the dataImport class to extract and preprocess text from the PDF file.
tagged_data = dataImport.importPdf(pdf_path)

# Initialize the Doc2Vec model with specific parameters.
# - vector_size: Dimensionality of the feature vectors.
# - window: The maximum distance between the current and predicted word within a sentence.
# - min_count: Ignores all words with total frequency lower than this.
# - workers: Number of worker threads to train the model (faster training with multicore machines).
# - epochs: Number of iterations (passes) over the corpus.
model = Doc2Vec(vector_size=50, window=5, min_count=5, workers=4, epochs=100)

# Build a vocabulary from the tagged data.
model.build_vocab(tagged_data)

# Train the Doc2Vec model on the tagged data.
# - total_examples: Total number of documents.
# - epochs: Number of iterations over the corpus.
model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)

# Save the model
model.save("jesc102_model.d2v")

