import os
import sys
import mlflow
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings

# Proje dizinini ekle
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.rag import get_rag_chain

# --- AYARLAR ---
EXPERIMENT_NAME = "Datasheet_RAG_v1"
RUN_NAME = "Baseline_Mistral_Chunk1000"
EVAL_DATASET = [
    # Datasheet'e uygun GERÇEK teknik sorular
    "What is the maximum clock frequency for the APB2 bus?",
    "Which pin is used for I2C1_SDA?",
    "What is the typical power consumption in Standby mode?"
]

# Jüri Modeli Ayarları
judge_llm = LangchainLLMWrapper(ChatMistralAI(model="mistral-small", temperature=0))
judge_embeddings = LangchainEmbeddingsWrapper(MistralAIEmbeddings())

faithfulness.llm = judge_llm
answer_relevancy.llm = judge_llm
answer_relevancy.embeddings = judge_embeddings

def run_experiment():
    # 1. MLflow Deneyini Ayarla
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    with mlflow.start_run(run_name=RUN_NAME):
        print(f"🧪 Deney Başlıyor: {RUN_NAME}")
        
        # 2. Parametreleri Logla (Bunlar ingest.py ve rag.py içindeki ayarların)
        # İleride bunları config dosyasından çekeceğiz
        mlflow.log_param("chunk_size", 1000)
        mlflow.log_param("chunk_overlap", 100)
        mlflow.log_param("model", "mistral-small")
        mlflow.log_param("k_retrieval", 3)

        # 3. Zinciri Yükle ve Soruları Sor
        chain = get_rag_chain()
        results = {"question": [], "answer": [], "contexts": []}

        print("🤖 Sorular soruluyor...")
        for q in EVAL_DATASET:
            response = chain(q)
            results["question"].append(q)
            results["answer"].append(response["answer"])
            results["contexts"].append([doc.page_content for doc in response["source_documents"]])

        # 4. Ragas ile Puanla
        print("⚖️  Puanlanıyor...")
        dataset = Dataset.from_dict(results)
        scores = evaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevancy],
            llm=judge_llm, 
            embeddings=judge_embeddings
        )

        # 5. Metrikleri MLflow'a Kaydet
        print(f"📈 Skorlar: {scores}")
        
        # Ortalama skorları hesapla (scores bir EvaluationResult objesi)
        df = scores.to_pandas()
        avg_faithfulness = df["faithfulness"].mean()
        avg_answer_relevancy = df["answer_relevancy"].mean()
        
        mlflow.log_metric("faithfulness", avg_faithfulness)
        mlflow.log_metric("answer_relevancy", avg_answer_relevancy)

        # 6. Sonuçları CSV yap ve Artifact olarak kaydet
        csv_path = "eval_results.csv"
        df.to_csv(csv_path, index=False)
        mlflow.log_artifact(csv_path)
        
        print("✅ Deney tamamlandı ve MLflow'a kaydedildi.")

if __name__ == "__main__":
    run_experiment()