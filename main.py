import io
import cv2
import torch
import tempfile
import numpy as np
import os

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from transformers import AutoImageProcessor, SiglipForImageClassification

MODEL_IDENTIFIER = "Ateeqq/ai-vs-human-image-detector"

MAX_IMAGE_SIZE = 10 * 1024 * 1024      # 10MB
MAX_VIDEO_SIZE = 50 * 1024 * 1024      # 50MB
MAX_VIDEO_SECONDS = 30                 # 30 секунд максимум

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.environ["TRANSFORMERS_CACHE"] = "./models_cache"

# explicitly request the fast processor to avoid the "slow image processor" warning
processor = AutoImageProcessor.from_pretrained(MODEL_IDENTIFIER, use_fast=True)
model = SiglipForImageClassification.from_pretrained(MODEL_IDENTIFIER)
model.to(device)
model.eval()

app = FastAPI(title="AI Detector API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # можно сузить позже
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "ok"}

# ---------------- IMAGE ----------------

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid image type")

    image_bytes = await file.read()

    if len(image_bytes) > MAX_IMAGE_SIZE:
        raise HTTPException(status_code=400, detail="Image too large (max 10MB)")

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Invalid image file")

    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        logits = model(**inputs).logits

    probs = torch.softmax(logits, dim=-1)
    idx = logits.argmax(-1).item()

    return {
        "label": model.config.id2label[idx],
        "confidence": round(probs[0, idx].item(), 4),
    }

# ---------------- VIDEO ----------------

@app.post("/predict-video")
async def predict_video(file: UploadFile = File(...)):

    if not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="Invalid video type")

    video_bytes = await file.read()

    if len(video_bytes) > MAX_VIDEO_SIZE:
        raise HTTPException(status_code=400, detail="Video too large (max 50MB)")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(video_bytes)
        video_path = tmp.name

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        os.remove(video_path)
        raise HTTPException(status_code=500, detail="Cannot open video")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    duration = total_frames / fps if fps > 0 else 0

    if duration > MAX_VIDEO_SECONDS:
        cap.release()
        os.remove(video_path)
        raise HTTPException(status_code=400, detail="Video too long (max 30 sec)")

    results = []
    frame_count = 0
    step = 10

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % step != 0:
            continue

        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        inputs = processor(images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            logits = model(**inputs).logits

        probs = torch.softmax(logits, dim=-1)
        idx = logits.argmax(-1).item()

        results.append({
            "label": model.config.id2label[idx],
            "confidence": probs[0, idx].item()
        })

        if len(results) >= 30:
            break

    cap.release()
    os.remove(video_path)

    ai = [r["confidence"] for r in results if r["label"] == "ai"]
    hum = [r["confidence"] for r in results if r["label"] == "hum"]

    ai_mean = float(np.mean(ai)) if ai else 0.0
    hum_mean = float(np.mean(hum)) if hum else 0.0

    return {
        "final_label": "ai" if ai_mean > hum_mean else "hum",
        "ai_score": round(ai_mean, 4),
        "human_score": round(hum_mean, 4),
        "frames": len(results),
    }