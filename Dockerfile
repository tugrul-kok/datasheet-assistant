# 1. Hafif bir Python imajı seç
FROM python:3.10-slim

# 2. Çalışma dizini oluştur
WORKDIR /app

# 3. Gerekli sistem kütüphanelerini yükle (Chromadb bazen build tools ister)
RUN apt-get update && apt-get install -y build-essential curl && rm -rf /var/lib/apt/lists/*

# 4. Kütüphaneleri kopyala ve yükle
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Kaynak kodları kopyala
COPY . .

# 6. Uygulamanın çalışacağı portu aç
EXPOSE 8000

# 7. Uygulamayı başlat
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]