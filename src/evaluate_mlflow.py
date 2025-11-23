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

# ARTIK get_rag_chain YERİNE ask_question KULLANIYORUZ
from src.rag import ask_question

# --- AYARLAR ---
EXPERIMENT_NAME = "Datasheet_RAG_MultiDoc"
RUN_NAME = "Mistral_Routing_Test"

# TEST SENARYOLARI (Soru + Hangi Dokümanda Aranacağı)
# Bu liste senin sisteminin "Routing" zekasını test eder.
EVAL_SCENARIOS = [
    # --- GRUP 1: STM32F407 (High Performance) ---
    {
        "question": "What is the maximum CPU frequency?",
        "doc_filter": "stm32f4.pdf",
        "expected_context_hint": "168 MHz"
    },
    {
        "question": "Which bus is connected to APB2?",
        "doc_filter": "stm32f4.pdf",
        "expected_context_hint": "high-speed"
    },
    {
        "question": "What is the function of pin PA9?",
        "doc_filter": "stm32f4.pdf",
        "expected_context_hint": "USART1"
    },
    
    # --- GRUP 2: STM32F103 (Blue Pill - Entry Level) ---
    {
        "question": "What is the Cortex core used in this microcontroller?",
        "doc_filter": "stm32f1.pdf",
        "expected_context_hint": "Cortex-M3"
    },
    {
        "question": "Is the PA10 pin 5V tolerant?",
        "doc_filter": "stm32f1.pdf",
        "expected_context_hint": "tolerant" 
    },
    {
        "question": "What is the flash memory size range?",
        "doc_filter": "stm32f1.pdf",
        "expected_context_hint": "64 or 128 Kbytes"
    },

    # --- GRUP 3: Quectel BG96 (Cellular IoT Modem) ---
    {
        "question": "Which LTE categories does the module support?",
        "doc_filter": "bg96.pdf",
        "expected_context_hint": "Cat M1"
    },
    {
        "question": "What is the power supply voltage range (VBAT)?",
        "doc_filter": "bg96.pdf",
        "expected_context_hint": "3.3V"
    },
    {
        "question": "Describe the PSM (Power Saving Mode) features.",
        "doc_filter": "bg96.pdf",
        "expected_context_hint": "power saving"
    },

    # --- GRUP 4: AUTO MODE (Routing Testi - En Kritiği) ---
    # Burada doc_filter='auto' veriyoruz, sistemin kendisinin F4 dokümanına gitmesini bekliyoruz.
    {
        "question": "What is the clock speed of STM32F407?",
        "doc_filter": "auto",
        "expected_context_hint": "168 MHz"
    },
    {
        "question": "Does BG96 support NB-IoT?",
        "doc_filter": "auto",
        "expected_context_hint": "NB-IoT"
    },

    # --- GRUP 5: NEGATİF TESTLER (Faithfulness Kontrolü) ---
    # Modelin "Bilmiyorum" demesi gereken sorular.
    {
        "question": "What is the price of STM32F407 in 2024?",
        "doc_filter": "stm32f4.pdf",
        "expected_context_hint": "cannot find" # Cevabın negatif olmasını bekliyoruz
    },
    {
        "question": "Does STM32F103 support WiFi connectivity?",
        "doc_filter": "stm32f1.pdf",
        "expected_context_hint": "cannot find" 
    }
]

# Jüri Modeli Ayarları
judge_llm = LangchainLLMWrapper(ChatMistralAI(model="mistral-small", temperature=0))
judge_embeddings = LangchainEmbeddingsWrapper(MistralAIEmbeddings())

faithfulness.llm = judge_llm
answer_relevancy.llm = judge_llm
answer_relevancy.embeddings = judge_embeddings

def run_experiment():
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    with mlflow.start_run(run_name=RUN_NAME):
        print(f"🧪 Deney Başlıyor: {RUN_NAME}")
        
        # Parametreleri Logla
        mlflow.log_param("strategy", "multi_doc_filtering")
        mlflow.log_param("test_size", len(EVAL_SCENARIOS))

        results = {
            "question": [],
            "answer": [],
            "contexts": [],
            "doc_filter": [] # Hangi filtreyle sorduğumuzu da kaydedelim
        }

        print("🤖 Sorular soruluyor (Multi-Doc)...")
        for scenario in EVAL_SCENARIOS:
            q = scenario["question"]
            f = scenario["doc_filter"]
            
            print(f"   👉 Soru: {q} | Filtre: {f}")
            
            # Yeni ask_question fonksiyonunu kullanıyoruz
            response = ask_question(query=q, doc_filter=f)
            
            results["question"].append(q)
            results["answer"].append(response["answer"])
            results["doc_filter"].append(f)
            
            # Context'leri listeye çevir
            context_list = [doc.page_content for doc in response["source_documents"]]
            results["contexts"].append(context_list)

        # Ragas için Dataset oluştur
        # Not: 'doc_filter' kolonunu Ragas kullanmaz ama Pandas dataframe'de analiz için tutarız.
        dataset = Dataset.from_dict(results)

        print("⚖️  Puanlanıyor...")
        scores = evaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevancy],
            llm=judge_llm, 
            embeddings=judge_embeddings
        )

        print(f"📈 Skorlar: {scores}")
        
        # Metrikleri Kaydet
        df = scores.to_pandas()
        
        # Ortalama skorlar
        mlflow.log_metric("avg_faithfulness", df["faithfulness"].mean())
        mlflow.log_metric("avg_relevancy", df["answer_relevancy"].mean())

        # Sonuçları Kaydet
        csv_path = "eval_results_multidoc.csv"
        df.to_csv(csv_path, index=False)
        mlflow.log_artifact(csv_path)
        
        print("✅ Multi-Doc Testi tamamlandı!")

if __name__ == "__main__":
    run_experiment()