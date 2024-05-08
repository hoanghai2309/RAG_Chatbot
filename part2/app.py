import os
# Disable parallelism in tokenizers to prevent potential issues
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langserve import add_routes
from llm_model import load_llm
from main import build_rag_chain, InputQA, OutputQA

# Initialize the language model with a temperature of 0.9
llm = load_llm('model/vinallama-7b-chat_q5_0.gguf')
# Define the directory for the generative AI documents
genai_docs = "data"

# Build the RAG chain for the generative AI documents
genai_chain = build_rag_chain(llm, data_dir=genai_docs, data_type="pdf")

# Initialize the FastAPI application
app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple api server using Langchain’s Runnable interfaces",
)

# Add CORS middleware to the application
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Define the routes for the application
@app.get("/check")
async def check():
    return {"status": "ok"}

@app.post("/generative_ai", response_model=OutputQA)
async def generative_ai(inputs: InputQA):
    answer = genai_chain.invoke(inputs.question)
    return {"answer": answer}

# Add routes for the playground
add_routes(app, genai_chain, playground_type="default", path="/generative_ai")


