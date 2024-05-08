from pydantic import BaseModel, Field
from file_loader import Loader
from vectorstore import VectorDB
from offline_rag import Offline_RAG

# Define a Pydantic model for the input question
class InputQA(BaseModel):
    question: str = Field(..., title="Question to ask the model")

# Define a Pydantic model for the output answer
class OutputQA(BaseModel):
    answer: str = Field(..., title="Answer from the model")

# Function to build the RAG chain
def build_rag_chain(llm, data_dir, data_type):
    # Load the documents from the directory
    doc_loaded = Loader(file_type=data_type).load_dir(data_dir, workers=2)
    # Get the retriever from the VectorDB
    retriever = VectorDB(documents=doc_loaded).get_retriever()
    # Get the RAG chain from the Offline_RAG
    rag_chain = Offline_RAG(llm).get_chain(retriever)
    # Return the RAG chain
    return rag_chain
