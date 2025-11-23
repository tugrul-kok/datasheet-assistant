import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

# Veri Yolu ve DB Ayarları
DATA_PATH = "data/"
DB_PATH = "chroma_db"

def ingest_data():
    # 1. PDF'leri Yükle
    print("📂 PDF'ler yükleniyor...")
    loader = PyPDFDirectoryLoader(DATA_PATH)
    documents = loader.load()
    
    # 2. Metni Parçala (Chunking)
    print("✂️  Metin parçalanıyor...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    chunks = text_splitter.split_documents(documents)
    
    # 3. Embedding ve Kayıt (Vektör DB)
    print("💾 Vektör veritabanına kaydediliyor...")
    embeddings = MistralAIEmbeddings()
    
    # Vektörleri diske kaydet
    Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory=DB_PATH
    )
    print(f"✅ İşlem tamam! {len(chunks)} parça vektörleştirildi.")

if __name__ == "__main__":
    ingest_data()