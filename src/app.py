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

# Use absolute path for static files to ensure it works in Docker
static_dir = project_root / "src" / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "project_root": str(project_root)}

class QueryRequest(BaseModel):
    query: str
    doc_type: str = "auto"  # Yeni parametre (Default: auto)

@app.get("/")
async def read_root():
    # Use absolute path to ensure it works in Docker
    index_path = project_root / "src" / "static" / "index.html"
    if not index_path.exists():
        raise FileNotFoundError(f"Index file not found at: {index_path}")
    return FileResponse(str(index_path))

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
    import os
    port = int(os.getenv("PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port)