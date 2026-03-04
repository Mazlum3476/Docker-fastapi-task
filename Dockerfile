FROM --platform=linux/arm64 python:3.9-slim

WORKDIR /app

# kütüp'leri kur
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# bütün dosyaları içeri atıyoruz
COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7001"]