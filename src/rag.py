import os
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

DB_PATH = "chroma_db"

# Global Modeller
embedding_function = MistralAIEmbeddings()
vector_db = Chroma(persist_directory=DB_PATH, embedding_function=embedding_function)
# Router için sıcaklık 0 olsun, karar net olsun
llm = ChatMistralAI(model="mistral-small", temperature=0.0)

# --- 1. ROUTER (KARAR MEKANİZMASI) ---
def semantic_router(query):
    """
    Sorunun hangi dokümanla ilgili olduğunu anlayan ajan.
    """
    print(f"🤔 Router Düşünüyor: '{query}'")
    
    router_template = """
    You are an expert intent classifier.
    Classify the user question into one of the following document keys:
    
    - stm32f4.pdf (Keywords: F4, F407, Discovery, 168MHz, DSP, Cortex-M4)
    - stm32f1.pdf (Keywords: F1, F103, Blue Pill, 72MHz, Cortex-M3)
    - bg96.pdf (Keywords: Modem, LTE, Cellular, NB-IoT, Cat M1, GNSS, Quectel)
    - stm32u5.pdf (Keywords: U5, Low Power, Cortex-M33)
    - auto (If the question is general or ambiguous)

    Examples:
    Q: What is the clock speed of F4? -> stm32f4.pdf
    Q: Does the modem support GPS? -> bg96.pdf
    Q: What is an interrupt? -> auto

    Question: {question}
    
    Return ONLY the filename (or 'auto'). Do not explain.
    """
    
    prompt = ChatPromptTemplate.from_template(router_template)
    chain = prompt | llm | StrOutputParser()
    
    try:
        route = chain.invoke({"question": query}).strip()
        print(f"👉 Router Kararı: {route}")
        # Bazen model "Filename: stm32f4.pdf" diyebilir, temizleyelim
        for key in ["stm32f4.pdf", "stm32f1.pdf", "bg96.pdf", "stm32u5.pdf"]:
            if key in route:
                return key
        return "auto"
    except Exception as e:
        print(f"Router Error: {e}")
        return "auto"

# --- 2. RAG CHAIN (DİNAMİK) ---
def get_rag_chain(doc_filter=None):
    # Ayarlar
    search_kwargs = {"k": 6}
    
    # Eğer filtre varsa onu uygula
    if doc_filter and doc_filter != "auto":
        source_path = f"data/{doc_filter}"
        search_kwargs["filter"] = {"source": source_path}
        print(f"🔍 Filtering Context: Only using {source_path}")
    else:
        print("🔍 Context: Global Search (No Filter)")

    retriever = vector_db.as_retriever(search_kwargs=search_kwargs)
    
    # Prompt - Biraz daha konuşkan hale getirdik (Relevancy için)
    template = """
    You are a Senior Embedded Systems Engineer. 
    Answer the question based ONLY on the provided context.
    
    Rules:
    1. Start directly with the answer. 
    2. If the info is in a table, mention it (e.g., "According to Table 4...").
    3. If the context is empty or irrelevant, say "I cannot find this specific information in the selected document."
    4. Be concise but complete.

    Context:
    {context}
    
    Question: {input}
    
    Answer:
    """
    prompt = ChatPromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain, retriever

# --- 3. ANA FONKSİYON ---
def ask_question(query, doc_filter="auto"):
    final_filter = doc_filter
    
    # Eğer kullanıcı "Auto" seçtiyse, Router devreye girsin
    if doc_filter == "auto":
        predicted_filter = semantic_router(query)
        # Router "auto" demezse, onun tahminini kullanalım
        if predicted_filter != "auto":
            final_filter = predicted_filter
    
    # RAG Zincirini çağır
    chain, retriever = get_rag_chain(final_filter)
    
    answer = chain.invoke(query)
    source_docs = retriever.invoke(query)
    
    return {
        "answer": answer,
        "source_documents": source_docs,
        "routed_to": final_filter # Debug için bunu da görelim
    }