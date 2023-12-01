import PyPDF2
import nltk
from nltk.tokenize import word_tokenize
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.corpus import stopwords

# This class method is responsible for downloading necessary NLTK resources,
# which include tokenization tools and a list of stopwords.
class embedding:

    # Download required NLTK resources, if not already downloaded
    def download_nltk_resources():
        nltk.download('punkt')  # For tokenization
        nltk.download('stopwords')  # For removing stop words

    # Ensure NLTK resources are downloaded before proceeding
    download_nltk_resources()

    # This method preprocesses a given sentence by tokenizing it, converting it to lowercase,
    # and removing any tokens that are not alphabetic or are considered stopwords.
    def preprocess_sentence(sentence):
        from nltk.corpus import stopwords
        stop_words = set(stopwords.words('english'))
        # Tokenize the sentence
        tokens = word_tokenize(sentence.lower())
        # Remove non-alphabetic characters and stopwords
        tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
        return tokens

    # This generator function opens a PDF and yields preprocessed text from each page as a TaggedDocument.
    # The TaggedDocument is suitable for training Doc2Vec models in Gensim.
    def extract_text_by_page(pdf_path):
        # Context manager to open and read the PDF file
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for i, page in enumerate(reader.pages):
                # Extract text from page and preprocess
                text = page.extract_text()
                sentences = nltk.sent_tokenize(text)
                # Preprocess each sentence and combine into one list of words for the page
                words = [embedding.preprocess_sentence(sentence) for sentence in sentences]
                words = [word for sublist in words for word in sublist]  # Flatten the list
                # Yielding a TaggedDocument for each page
                yield TaggedDocument(words=words, tags=[str(i)])


    
