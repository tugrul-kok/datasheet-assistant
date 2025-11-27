# 🧠 Datasheet RAG Assistant with MLOps Pipeline

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Mistral AI](https://img.shields.io/badge/AI-Mistral%20Small-orange)
![MLOps](https://img.shields.io/badge/MLOps-CI%2FCD-green)
![Docker](https://img.shields.io/badge/Container-Docker-blue)

## 🚀 Project Overview
This project is a production-ready **Retrieval-Augmented Generation (RAG)** system designed to assist embedded systems engineers in navigating complex datasheets (e.g., STM32, BME280).

Unlike standard RAG demos, this project implements a full **MLOps lifecycle**, including automated data ingestion, drift detection, experiment tracking with MLflow, and CI/CD deployment via GitHub Actions.

### 🔗 Live Demo: [datasheet.tugrul.app](https://datasheet.tugrul.app)

---

## 🏗 Architecture & MLOps Workflow

The system is deployed on a Hetzner Cloud VPS using Docker containers.

1.  **Data Ingestion:** PDF datasheets are parsed, chunked, and embedded into a ChromaDB vector store.
2.  **Semantic Router:** An intelligent LLM-based routing system that automatically determines which datasheet to search based on the user's query (e.g., routes "F407" questions to STM32F4, "modem" questions to BG96).
3.  **RAG Pipeline:** - **Model:** `mistral-small` (via Mistral API).
    - **Embeddings:** `mistral-embed`.
    - **Framework:** LangChain & FastAPI.
4.  **Evaluation (CI):** Every code push triggers an automated evaluation pipeline using **Ragas** (Faithfulness & Relevancy metrics) to prevent regression.
5.  **Deployment (CD):** GitHub Actions automatically builds the Docker container and deploys it to the production server upon passing tests.

---

## 💡 Case Study: Solving "Datasheet Fatigue"

### 🔴 The Problem: The Embedded Engineer's Nightmare
Working with embedded systems (like STM32 or NXP) requires constant reference to 2000+ page datasheets.
- **Inefficiency:** Engineers spend up to 30% of development time just searching for register addresses or pin functions.
- **The "Table" Issue:** Standard LLMs and search tools struggle to interpret **complex electrical tables**, often losing the connection between a Pin Name (e.g., PA9) and its Alternate Function.

### 🟢 The Solution: Domain-Specific RAG Optimization
We didn't just build a chatbot; we built an engineering tool.
1.  **Table-Aware Ingestion:** We optimized the chunking strategy (2000 tokens with overlap) specifically to keep table rows and headers in the same semantic context.
2.  **Semantic Routing:** An intelligent document router uses Mistral to analyze queries and automatically route them to the most relevant datasheet. For example:
    - Questions mentioning "F407" or "F4" → Routes to STM32F4 datasheet
    - Questions about "modem", "LTE", or "cellular" → Routes to BG96 datasheet
    - Generic questions → Searches across all documents
3.  **Feedback Loop:** We used **MLOps best practices** (MLflow & Ragas) to measure the model's performance on technical questions, improving "Faithfulness" from failing grades to a perfect 1.00 score.
---

## 🤖 Why Mistral AI?

This project strictly utilizes **Mistral AI** models (`mistral-small` and `mistral-embed`) for the following reasons:

1.  **Efficiency/Cost Ratio:** `mistral-small` offers comparable reasoning capabilities to larger proprietary models (like GPT-3.5/4) but with significantly lower latency and cost, which is crucial for real-time industrial assistants.
2.  **Context Handling:** Mistral's sliding window attention mechanism handles the dense, technical context of engineering datasheets better than older transformer architectures.
3.  **European Sovereignty:** As an EU-based engineer, leveraging a GDPR-compliant, high-performance European LLM is a strategic choice for enterprise data privacy.

---

## 🛠️ Setup & Local Development

```bash
# 1. Clone the repo
git clone [https://github.com/tugrul-kok/datasheet-assistant.git](https://github.com/tugrul-kok/datasheet-assistant.git)

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up environment variables
# Create a .env file with: MISTRAL_API_KEY=...

# 4. Run the pipeline
python src/ingest.py  # Process Data
uvicorn src.app:app --reload # Start Server
```

## 🧠 Key Features

### Semantic Routing
The system includes an intelligent semantic router that automatically determines which datasheet to search based on the user's query. When users select "Auto" mode, the router analyzes the question and routes it to the most relevant document:

- **STM32F4**: High-performance, DSP, FPU, 168MHz, Cortex-M4
- **STM32F1**: Entry-level, Blue Pill, 72MHz, Cortex-M3
- **STM32U5**: Ultra-low power, Cortex-M33, Security, TrustZone
- **BG96**: Cellular modem, LTE, Cat M1, NB-IoT, GNSS

Users can also manually select a specific document or use "Auto" mode for intelligent routing.