import os
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

DB_PATH = "chroma_db"

def get_rag_chain():
    # 1. Kayıtlı Veritabanını Bağla
    embedding_function = MistralAIEmbeddings()
    vector_db = Chroma(persist_directory=DB_PATH, embedding_function=embedding_function)
    
    # 2. Retriever (Getirici) Ayarla
    retriever = vector_db.as_retriever(search_kwargs={"k": 3}) # En alakalı 3 parçayı getir
    
    # 3. LLM Tanımla
    llm = ChatMistralAI(model="mistral-small", temperature=0)
    
    # 4. Prompt Template Oluştur
    prompt = ChatPromptTemplate.from_template("""
    Aşağıdaki bağlamı kullanarak soruyu yanıtlayın. 
    Eğer cevabı bilmiyorsanız, bilmediğinizi söyleyin, cevap uydurmayın.
    
    Bağlam: {context}
    
    Soru: {input}
    
    Cevap:
    """)
    
    # 5. RAG Chain Oluştur (LangChain 1.0+ için)
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    # Kaynak belgeleri de döndürmek için wrapper fonksiyon
    def rag_with_sources(query):
        # Belgeleri al
        docs = retriever.invoke(query)
        
        # Context'i formatla
        context = format_docs(docs)
        
        # Prompt'u hazırla ve LLM'e gönder
        messages = prompt.format_messages(context=context, input=query)
        response = llm.invoke(messages)
        answer = response.content if hasattr(response, 'content') else str(response)
        
        return {
            "answer": answer,
            "source_documents": docs
        }
    
    return rag_with_sources