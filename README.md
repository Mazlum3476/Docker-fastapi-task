# Object Detection Service (FastAPI & Docker)

This repository contains the deployment of a **ResNet50-based** object detection model trained in the previous task. The model is served using **FastAPI** and containerized with **Docker** for cross-platform compatibility.

## 🚀 Features
- **Model:** PyTorch-based ResNet50 (Fine-tuned for car detection).
- **Service:** High-performance inference API built with FastAPI.
- **Docker:** Optimized image configured to run on ARM-based MacOS devices.
- **Output:** Prediction results are returned in structured JSON format.

## 📁 Project Structure
- `main.py`: FastAPI application logic and inference pipeline.
- `model.py`: Model architecture definition.
- `model_weights.pth`: Trained model weights (state dictionary).
- `Dockerfile`: Docker image configuration for ARM64.
- `docker-compose.yml`: Orchestration to run the service on port 7001.
- `requirements.txt`: Necessary Python dependencies.

## 🛠️ Installation & Usage

Ensure you have **Docker Desktop** installed on your machine. Follow these steps to run the service:

1. Clone the repository:
```bash
git clone [https://github.com/Mazlum3476/Docker-fastapi-task.git](https://github.com/Mazlum3476/Docker-fastapi-task.git)
```

Navigate to the project directory and build the container:
```docker-compose up --build ```

Access the service:
Base URL: http://localhost:7001
API Documentation (Swagger): http://localhost:7001/docs

API Response Example
When an image is uploaded to the /predict endpoint, the service returns:
```
{
  "filename": "car_image.jpg",
  "prediction": "Araba",
  "confidence": 0.9237,
  "status": "success"
}
```
Developed by: Mazlum Dağcı
