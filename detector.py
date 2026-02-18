import os
import torch
import cv2
import numpy as np
from typing import Optional
from PIL import Image
from transformers import AutoImageProcessor, SiglipForImageClassification

MODEL_ID = "Ateeqq/ai-vs-human-image-detector"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model/processor will be initialized on first use to avoid long import time
processor: Optional[AutoImageProcessor] = None
model: Optional[SiglipForImageClassification] = None

def _load_model() -> None:
    """Lazy load the transformer processor and model."""
    global processor, model
    if processor is None or model is None:
        processor = AutoImageProcessor.from_pretrained(MODEL_ID)
        model = SiglipForImageClassification.from_pretrained(MODEL_ID)
        model.to(device)
        model.eval()



def predict_image(image_path: str) -> dict:
    """Return prediction for a single image file.

    Raises FileNotFoundError or PIL.UnidentifiedImageError if the file
    cannot be opened.
    """
    _load_model()
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        logits = model(**inputs).logits

    probs = torch.softmax(logits, dim=-1)
    idx = logits.argmax(-1).item()

    return {
        "label": model.config.id2label[idx],
        "confidence": float(probs[0, idx])
    }


def predict_video(video_path: str, step: int = 10) -> dict:
    """Sample frames from a video and return an aggregated prediction.

    The `step` parameter controls how often (by frame count) we sample.
    """
    _load_model()

    if not os.path.exists(video_path):
        raise FileNotFoundError(video_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")

    results = []
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        count += 1
        if count % step != 0:
            continue

        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        inputs = processor(images=img, return_tensors="pt").to(device)

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

    ai_score = float(np.mean(ai)) if ai else 0.0
    hum_score = float(np.mean(hum)) if hum else 0.0

    return {
        "final_label": "ai" if ai_score > hum_score else "hum",
        "ai_score": ai_score,
        "human_score": hum_score,
        "frames": len(results)
    }


if __name__ == "__main__":
    # simple command‑line test harness
    import argparse

    parser = argparse.ArgumentParser(description="Run detector on an image or video")
    parser.add_argument("path", help="path to image or video file")
    parser.add_argument("--video", action="store_true", help="treat path as video")
    args = parser.parse_args()

    if args.video:
        print(predict_video(args.path))
    else:
        print(predict_image(args.path))
