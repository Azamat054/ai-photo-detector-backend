import io
import os
import cv2
import torch
import asyncio
import tempfile
import numpy as np

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from transformers import AutoProcessor, SiglipForImageClassification

MODEL_IDENTIFIER = "Ateeqq/ai-vs-human-image-detector"

MAX_IMAGE_SIZE = 10 * 1024 * 1024
MAX_VIDEO_SIZE = 50 * 1024 * 1024
MAX_VIDEO_SECONDS = 30

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.environ["TRANSFORMERS_CACHE"] = "./models_cache"

# ---------- LAZY LOAD ----------
processor = None
model = None
_model_lock = asyncio.Lock()


async def get_model():
    global processor, model

    if processor is not None and model is not None:
        return processor, model

    async with _model_lock:
        if processor is None or model is None:
            proc = AutoProcessor.from_pretrained(MODEL_IDENTIFIER)
            mdl = SiglipForImageClassification.from_pretrained(
                MODEL_IDENTIFIER
            )
            mdl.to(device)
            mdl.eval()

            processor = proc
            model = mdl

    return processor, model


app = FastAPI(title="AI Detector API")

# ---------- ✅ CORS FIX (ВАЖНО) ----------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://ai-photo-detector-mobile.vercel.app",
        "http://localhost:19006",
        "http://localhost:8081",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"status": "ok"}


# ---------- IMAGE ----------
@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid image type")

    image_bytes = await file.read()

    if len(image_bytes) > MAX_IMAGE_SIZE:
        raise HTTPException(status_code=400, detail="Image too large")

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Invalid image")

    processor, model = await get_model()

    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        logits = model(**inputs).logits

    probs = torch.softmax(logits, dim=-1)
    idx = logits.argmax(-1).item()

    return {
        "label": model.config.id2label[idx],
        "confidence": float(probs[0, idx]),
    }


# ---------- VIDEO ----------
@app.post("/predict-video")
async def predict_video(file: UploadFile = File(...)):
    if not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="Invalid video type")

    video_bytes = await file.read()

    if len(video_bytes) > MAX_VIDEO_SIZE:
        raise HTTPException(status_code=400, detail="Video too large")

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
        raise HTTPException(status_code=400, detail="Video too long")

    processor, model = await get_model()

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
            "confidence": float(probs[0, idx]),
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
        "ai_score": ai_mean,
        "human_score": hum_mean,
        "frames": len(results),
    }


# ---------- LOCAL RUN ----------
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)