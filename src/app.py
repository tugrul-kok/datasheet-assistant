from fastapi import FastAPI
from pydantic import BaseModel
from src.rag import get_rag_chain

app = FastAPI(title="MLOps RAG Chatbot")

# Zinciri bir kere başlat
chain = get_rag_chain()

class QueryRequest(BaseModel):
    query: str

@app.post("/chat")
def chat(request: QueryRequest):
    response = chain(request.query)
    
    return {
        "answer": response["answer"],
        "sources": [doc.page_content[:100] + "..." for doc in response["source_documents"]]
    }

# Çalıştırmak için terminale: uvicorn src.app:app --reload