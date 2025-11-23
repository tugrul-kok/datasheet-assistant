import os
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

DB_PATH = "chroma_db"

# Global değişkenler (Her istekte tekrar yüklememek için)
embedding_function = MistralAIEmbeddings()
vector_db = Chroma(persist_directory=DB_PATH, embedding_function=embedding_function)
llm = ChatMistralAI(model="mistral-small", temperature=0.0)

def get_rag_chain(doc_filter=None):
    """
    doc_filter: 'stm32f4.pdf' gibi dosya adı veya 'auto' olabilir.
    """
    
    # 1. Filtreleme Mantığı (Metadata Filtering)
    search_kwargs = {"k": 6}
    
    if doc_filter and doc_filter != "auto":
        # ChromaDB'de kaynaklar genellikle "data/dosyaadi.pdf" olarak saklanır
        # Bu yüzden filtreyi tam yola göre yapıyoruz
        source_path = f"data/{doc_filter}"
        search_kwargs["filter"] = {"source": source_path}
        print(f"🔍 Filtering Context: Only using {source_path}")
    else:
        print("🔍 Context: Auto (Searching all docs)")

    # 2. Retriever'ı Dinamik Oluştur
    retriever = vector_db.as_retriever(search_kwargs=search_kwargs)
    
    # 3. Prompt (Negatif Örnekli ve Otoriter)
    template = """
    You are a Senior Embedded Systems Engineer. Answer based ONLY on the provided context.
    
    --- EXAMPLES ---
    Q: What is the alternate function of PA9?
    A: According to [Table 9], PA9 corresponds to USART1_TX.
    
    Q: What is the price?
    A: I cannot find pricing info in the datasheet.
    --- END EXAMPLES ---

    Rules:
    1. If the context is empty or irrelevant, say "I cannot find this information in the selected document."
    2. Be precise.
    
    Context:
    {context}
    
    Question: {input}
    
    Answer:
    """
    prompt = ChatPromptTemplate.from_template(template)

    # 4. Chain Oluştur (LCEL Formatı - Daha Modern)
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    # Kaynakları da döndürmek için özel bir yapı döndürüyoruz
    return rag_chain, retriever

def ask_question(query, doc_filter="auto"):
    chain, retriever = get_rag_chain(doc_filter)
    
    # Cevabı al
    answer = chain.invoke(query)
    
    # Kaynakları manuel çek (Chain içinde kaybolmasın diye)
    source_docs = retriever.invoke(query)
    
    return {
        "answer": answer,
        "source_documents": source_docs
    }