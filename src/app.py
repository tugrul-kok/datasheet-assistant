from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from src.rag import get_rag_chain
import os

app = FastAPI(title="MLOps RAG Chatbot")

# Statik dosyaları bağla (CSS, JS, HTML için)
app.mount("/static", StaticFiles(directory="src/static"), name="static")

# Zinciri başlat
chain = get_rag_chain()

class QueryRequest(BaseModel):
    query: str

# Ana Sayfa (HTML'i döndür)
@app.get("/")
async def read_root():
    return FileResponse('src/static/index.html')

@app.post("/chat")
def chat(request: QueryRequest):
    response = chain(request.query)
    
    return {
        "answer": response["answer"],
        # Kaynakları biraz temizleyelim
        "sources": [doc.page_content[:100] + "..." for doc in response["source_documents"]]
    }

# Çalıştırmak için terminale: uvicorn src.app:app --reload