import io
import cv2
import torch
import tempfile
import numpy as np
import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image, UnidentifiedImageError
from transformers import AutoImageProcessor, SiglipForImageClassification
from fastapi.middleware.cors import CORSMiddleware

MODEL_IDENTIFIER = "Ateeqq/ai-vs-human-image-detector"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set model cache directory for faster loading
os.environ['TRANSFORMERS_CACHE'] = './models_cache'

try:
    processor = AutoImageProcessor.from_pretrained(MODEL_IDENTIFIER)
    model = SiglipForImageClassification.from_pretrained(MODEL_IDENTIFIER)
    model.to(device)
    model.eval()
except Exception as e:
    print(f"Error loading model: {e}")
    raise

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Application startup")
    yield
    print("Application shutdown")

app = FastAPI(title="AI Video Detector", lifespan=lifespan)

origins = [
    "http://localhost:8081",  # React фронт
    "http://127.0.0.1:3000",  # альтернативный вариант
    # можно добавить другие домены или "*" для всех
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,      # разрешённые домены
    allow_credentials=True,
    allow_methods=["*"],        # разрешённые методы GET, POST и т.д.
    allow_headers=["*"],        # разрешённые заголовки
)

@app.get("/")
def root():
    return {"status": "API is running"}


@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File is not an image")

    image_bytes = await file.read()
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Cannot open image")

    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        logits = model(**inputs).logits

    probs = torch.softmax(logits, dim=-1)
    idx = logits.argmax(-1).item()

    return {
        "filename": file.filename,
        "label": model.config.id2label[idx],
        "confidence": round(probs[0, idx].item(), 4),
        "scores": {
            model.config.id2label[i]: round(probs[0, i].item(), 4)
            for i in model.config.id2label
        }
    }


@app.post("/predict-video")
async def predict_video(file: UploadFile = File(...)):
    if not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="Not a video")

    # save video to temp file
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(await file.read())
            video_path = tmp.name
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save video: {e}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise HTTPException(status_code=500, detail="Cannot open video file")

    frame_results = []
    frame_count = 0
    step = 10  # 1 кадр из 10

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % step != 0:
            continue

        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
        except Exception:
            continue  # пропускаем проблемный кадр

        try:
            inputs = processor(images=image, return_tensors="pt").to(device)
            with torch.no_grad():
                logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)
            idx = logits.argmax(-1).item()
            frame_results.append({
                "label": model.config.id2label[idx],
                "confidence": probs[0, idx].item()
            })
        except Exception:
            continue  # пропускаем кадр если модель выдала ошибку

    cap.release()

    if not frame_results:
        raise HTTPException(status_code=400, detail="No usable frames extracted from video")

    # aggregate results
    ai_scores = [r["confidence"] for r in frame_results if r["label"] == "ai"]
    hum_scores = [r["confidence"] for r in frame_results if r["label"] == "hum"]

    ai_score = np.mean(ai_scores) if ai_scores else 0.0
    hum_score = np.mean(hum_scores) if hum_scores else 0.0

    final_label = "ai" if ai_score > hum_score else "hum"

    return {
        "filename": file.filename,
        "frames_used": len(frame_results),
        "final_label": final_label,
        "ai_score": round(float(ai_score), 4),
        "human_score": round(float(hum_score), 4)
    }
