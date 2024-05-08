from typing import Union
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Define the VectorDB class
class VectorDB:
    # Initialize the class
    def __init__(self, documents=None, vector_db: Union[Chroma, FAISS] = Chroma, embedding=HuggingFaceEmbeddings()) -> None:
        self.vector_db = vector_db  # Set the vector database
        self.embedding = embedding  # Set the embedding
        self.db = self._build_db(documents)  # Build the database

    # Method to build the database
    def _build_db(self, documents):
        db = self.vector_db.from_documents(documents=documents, embedding=self.embedding)  # Create the database from documents
        return db  # Return the database

    # Method to get the retriever
    def get_retriever(self, search_type: str = "similarity", search_kwargs: dict = {"k": 3}):
        retriever = self.db.as_retriever(search_type=search_type, search_kwargs=search_kwargs)  # Get the retriever
        return retriever  # Return the retriever
