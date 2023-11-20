from data_import import dataImport
import config
from gensim.models.doc2vec import Doc2Vec, TaggedDocument


pdf_path = config.ai_book_path

tagged_data = dataImport.importPdf(pdf_path)

model = Doc2Vec(vector_size=50, window=5, min_count=5, workers=4, epochs=100)
model.build_vocab(tagged_data)
model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)

# Save the model
model.save("jesc102_model.d2v")

