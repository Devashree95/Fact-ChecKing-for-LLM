from langchain.text_splitter import RecursiveCharacterTextSplitter

class preprocess:

    # This function divides a section of text into smaller chunks.
    def chunk_section(section, chunk_size, chunk_overlap):
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", " ", ""], # Define separators for splitting text.
            chunk_size=chunk_size,              # Maximum size of each chunk.
            chunk_overlap=chunk_overlap,        # Number of characters to overlap between chunks.
            length_function=len)
        # Use the text splitter to divide the input text into chunks.
        chunks = text_splitter.create_documents(
            texts=[section["text"]],
            metadatas=[{"source": section["source"]}])
        # Return a list of dictionaries, each containing a chunk of text and its associated metadata.
        return [{"text": chunk.page_content, "source": chunk.metadata["source"]} for chunk in chunks]
