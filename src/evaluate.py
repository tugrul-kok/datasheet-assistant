import os
import sys
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings

# Import path'i düzelt - proje kök dizinini ekle
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.rag import get_rag_chain

# 1. Jüri Modellerini Ayarla (Mistral Kullanacağız)
# Değerlendirmeyi yapan modelin biraz zeki olması gerekir, Mistral Small/Large iyidir.
judge_llm = LangchainLLMWrapper(ChatMistralAI(model="mistral-small", temperature=0))
judge_embeddings = LangchainEmbeddingsWrapper(MistralAIEmbeddings())

# Metriklere jüriyi tanıt
faithfulness.llm = judge_llm
answer_relevancy.llm = judge_llm
answer_relevancy.embeddings = judge_embeddings

def run_evaluation():
    print("🚀 Değerlendirme başlıyor...")
    
    # 2. Test Verisi (Bunu kendi PDF içeriğine göre MUTLAKA değiştir)
    # Gerçek hayatta bu "Golden Dataset" olarak dışarıdan yüklenir.
    test_questions = [
        "Tuğrul Kök hangi üniversiteden mezun olmuştur?",
        "Tuğrul'un uzmanlık alanları nelerdir?",
        "Madlen şirketinde hangi projeyi geliştirmiştir?" 
    ]
    
    # RAG Zincirini Yükle
    chain = get_rag_chain()
    
    results = {
        "question": [],
        "answer": [],
        "contexts": [],  # Ragas "contexts" (liste) bekler
    }

    # 3. Soruları Chatbot'a Sor ve Cevapları Topla
    print("🤖 Sorular chatbot'a soruluyor...")
    for q in test_questions:
        response = chain(q)
        
        results["question"].append(q)
        results["answer"].append(response["answer"])
        
        # Context'leri string listesi haline getir
        context_list = [doc.page_content for doc in response["source_documents"]]
        results["contexts"].append(context_list)

    # 4. Veriyi Dataset Formatına Çevir
    dataset = Dataset.from_dict(results)

    # 5. Ragas ile Puanla
    print("⚖️  Ragas puanlaması yapılıyor (Bu biraz sürebilir)...")
    scores = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy],
        llm=judge_llm, 
        embeddings=judge_embeddings
    )

    # 6. Sonuçları Göster ve Kaydet
    df = scores.to_pandas()
    print("\n📊 Değerlendirme Sonuçları:")
    print(df[["user_input", "faithfulness", "answer_relevancy"]])
    
    # Ortalama skoru yazdır
    print("\n📈 Ortalama Skorlar:")
    print(scores)
    
    # İstersen CSV olarak kaydet (MLOps'ta bu dosya versiyonlanır)
    df.to_csv("evaluation_results.csv", index=False)

if __name__ == "__main__":
    run_evaluation()