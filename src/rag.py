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
    
    # 2. Retriever (Getirici) Ayarla - DEĞİŞİKLİK BURADA
    # k=3 yetersiz kalıyordu, k=6 yaparak modele daha fazla "çevre bilgisi" veriyoruz.
    retriever = vector_db.as_retriever(search_kwargs={"k": 6})
    
    # 3. LLM Tanımla
    llm = ChatMistralAI(model="mistral-small", temperature=0.1) # Biraz yaratıcılık için 0.1
    
    # 4. Prompt Template Oluştur (PROMPT ENGINEERING)
    # Modele bir "Persona" veriyoruz ve formatı zorluyoruz.
    template = """
    You are a Senior Embedded Systems Engineer assisting a developer.
    Use the following context to answer the question accurately and concisely.
    
    --- EXAMPLES OF GOOD ANSWERS ---
    
    Question: What is the alternate function of PA9?
    Answer: According to the datasheet table [Table 9], Pin PA9 corresponds to the alternate function USART1_TX. It is 5V tolerant.
    
    Question: What is the typical current consumption in Stop mode?
    Answer: Based on the electrical characteristics section, the typical current consumption in Stop mode is 14 µA when VDD is 3.3V and 25°C.
    
    --- END OF EXAMPLES ---

    Rules:
    1. Mimic the structure of the examples above.
    2. Explicitly mention the table or section name if visible in the context.
    3. If the answer is not in the context, say "I cannot find this specific information."
    4. Do not make up information.

    Context:
    {context}
    
    Question: {input}
    
    Detailed Answer:
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # 5. RAG Chain Oluştur
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    def rag_with_sources(query):
        docs = retriever.invoke(query)
        context = format_docs(docs)
        messages = prompt.format_messages(context=context, input=query)
        response = llm.invoke(messages)
        answer = response.content if hasattr(response, 'content') else str(response)
        
        return {
            "answer": answer,
            "source_documents": docs
        }
    
    return rag_with_sources