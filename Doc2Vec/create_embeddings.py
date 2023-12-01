import PyPDF2
import nltk
from nltk.tokenize import word_tokenize
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.corpus import stopwords

class embedding:

    # Download required NLTK resources, if not already downloaded
    def download_nltk_resources():
        nltk.download('punkt')  # For tokenization
        nltk.download('stopwords')  # For removing stop words

    # Ensure NLTK resources are downloaded before proceeding
    download_nltk_resources()

    def preprocess_sentence(sentence):
        from nltk.corpus import stopwords
        stop_words = set(stopwords.words('english'))
        # Tokenize the sentence
        tokens = word_tokenize(sentence.lower())
        # Remove non-alphabetic characters and stopwords
        tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
        return tokens

    def extract_text_by_page(pdf_path):
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for i, page in enumerate(reader.pages):
                # Extract text from page and preprocess
                text = page.extract_text()
                sentences = nltk.sent_tokenize(text)
                # Preprocess each sentence and combine into one list of words for the page
                words = [embedding.preprocess_sentence(sentence) for sentence in sentences]
                words = [word for sublist in words for word in sublist]  # Flatten the list
                yield TaggedDocument(words=words, tags=[str(i)])


    
