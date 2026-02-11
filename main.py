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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.environ["TRANSFORMERS_CACHE"] = "./models_cache"

processor = AutoImageProcessor.from_pretrained(MODEL_IDENTIFIER)
model = SiglipForImageClassification.from_pretrained(MODEL_IDENTIFIER)
model.to(device)
model.eval()

app = FastAPI(title="AI Detector API")

# ✅ CORS — ГЛАВНОЕ ИСПРАВЛЕНИЕ
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
        raise HTTPException(status_code=400, detail="Not an image")

    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Invalid image")

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
        raise HTTPException(status_code=400, detail="Not a video")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(await file.read())
        video_path = tmp.name

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise HTTPException(status_code=500, detail="Cannot open video")

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

    ai = [r["confidence"] for r in results if r["label"] == "ai"]
    hum = [r["confidence"] for r in results if r["label"] == "hum"]

    return {
        "final_label": "ai" if np.mean(ai) > np.mean(hum) else "hum",
        "ai_score": round(float(np.mean(ai)) if ai else 0, 4),
        "human_score": round(float(np.mean(hum)) if hum else 0, 4),
        "frames": len(results),
    }
