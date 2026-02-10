import torch
import cv2
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, SiglipForImageClassification

MODEL_ID = "Ateeqq/ai-vs-human-image-detector"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processor = AutoImageProcessor.from_pretrained(MODEL_ID)
model = SiglipForImageClassification.from_pretrained(MODEL_ID)
model.to(device)
model.eval()


def predict_image(image_path: str):
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


def predict_video(video_path: str, step=10):
    cap = cv2.VideoCapture(video_path)

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

    ai_score = np.mean(ai) if ai else 0
    hum_score = np.mean(hum) if hum else 0                  

    return {
        "final_label": "ai" if ai_score > hum_score else "hum",
        "ai_score": ai_score,
        "human_score": hum_score,
        "frames": len(results)
    }
