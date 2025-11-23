from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os
import sys
from pathlib import Path

# Proje root'unu Python path'ine ekle
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import'u düzeltilmiş path ile yap
try:
    from src.rag import ask_question
except ImportError:
    # Eğer src.rag çalışmazsa, doğrudan rag'ı dene
    from rag import ask_question

app = FastAPI(title="Datasheet Assistant API")

app.mount("/static", StaticFiles(directory="src/static"), name="static")

class QueryRequest(BaseModel):
    query: str
    doc_type: str = "auto"  # Yeni parametre (Default: auto)

@app.get("/")
async def read_root():
    return FileResponse('src/static/index.html')

@app.post("/chat")
def chat(request: QueryRequest):
    response = ask_question(request.query, request.doc_type)
    
    # Hangi dokümana yönlendiğini cevapta gösterelim (Debug için harika)
    routing_info = ""
    if request.doc_type == "auto" and "routed_to" in response:
        routing_info = f"\n\n(🤖 Auto-routed to: {response['routed_to']})"
    
    return {
        "answer": response["answer"] + routing_info,
        "sources": [f"[{os.path.basename(doc.metadata.get('source', 'Unknown'))}] " + doc.page_content[:100] + "..." for doc in response["source_documents"]]
    }

# Uygulamayı doğrudan çalıştırmak için
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)