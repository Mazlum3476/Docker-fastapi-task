import os
import torch
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from torchvision import transforms
from model import ObjectDetector

app = FastAPI(title="DigiNova Object Detection API")


DEVICE = torch.device("cpu")


model = ObjectDetector(num_classes=2)


MODEL_PATH = "/app/model_weights.pth"

if not os.path.exists(MODEL_PATH):
    # Eğer üstteki yol hata verirse hata ayıklama için
    MODEL_PATH = "model_weights.pth"

try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    print(f"BAŞARILI: Model ağırlıkları {MODEL_PATH} üzerinden yüklendi.")
except Exception as e:
    print(f"HATA: Model yüklenemedi! Detay: {e}")


transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.get("/")
async def root():
    return {"message": "DigiNova Nesne Tespiti Servisi Çalışıyor. Test için /docs adresine gidin."}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Sadece görsel dosyalarını kabul et
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Lütfen geçerli bir görsel dosyası yükleyin.")

    try:
        
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("Görsel decode edilemedi.")
            
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        
        img_tensor = transform(img_rgb).unsqueeze(0).to(DEVICE)
        
       
        with torch.no_grad():
            _, class_logits = model(img_tensor)
            
        
        probabilities = torch.nn.functional.softmax(class_logits, dim=1).squeeze()
        max_prob, label_idx = torch.max(probabilities, dim=0)
        
        
        class_names = {0: "Arka Plan / Obje Yok", 1: "Araba"}
        predicted_label = class_names.get(label_idx.item(), "Bilinmeyen")
        
        # İstenen JSON formatında yanıt dön
        return {
            "filename": file.filename,
            "prediction": predicted_label,
            "confidence": round(float(max_prob.item()), 4),
            "status": "success"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"İşlem sırasında hata oluştu: {str(e)}")