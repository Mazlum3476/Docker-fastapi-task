FROM --platform=linux/arm64 python:3.9-slim

WORKDIR /app

# Önce kütüphaneleri kuruyoruz
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Şimdi bütün dosyaları (model_weights.pth dahil) içeri atıyoruz
COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7001"]