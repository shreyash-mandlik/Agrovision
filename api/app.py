import os
import json
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io

# ── TensorFlow import (suppress logs) ──────────────────────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

# ── App setup ──────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)  # Allow all origins so your HTML frontend can call this API

# ── Config ─────────────────────────────────────────────────────────────────
MODEL_PATH      = "agrovision_v2.h5"
CLASS_NAMES_PATH = "class_names.json"
IMG_SIZE        = (224, 224)
TOP_K           = 3  # Return top 3 predictions

# ── Load model and class names once at startup ─────────────────────────────
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded.")

with open(CLASS_NAMES_PATH, "r") as f:
    class_names = json.load(f)
print(f"Classes loaded: {len(class_names)} classes")

# ── Treatment recommendations per disease ──────────────────────────────────
# Add more as needed — keyed on class name (case-insensitive partial match)
TREATMENTS = {
    "healthy": {
        "type": "healthy",
        "severity": "None",
        "treatment": "Plant looks healthy! Maintain current care routine.",
        "prevention": "Continue regular watering, fertilization, and monitoring."
    },
    "early_blight": {
        "type": "disease",
        "severity": "Medium",
        "treatment": "Apply Mancozeb 2.5g/L or Chlorothalonil fungicide. Remove infected leaves.",
        "prevention": "Avoid overhead irrigation. Rotate crops. Use disease-free seeds."
    },
    "late_blight": {
        "type": "disease",
        "severity": "High",
        "treatment": "Spray Metalaxyl 2g/L immediately. Remove and destroy infected plants.",
        "prevention": "Use resistant varieties. Avoid planting in wet/humid conditions."
    },
    "leaf_spot": {
        "type": "disease",
        "severity": "Medium",
        "treatment": "Apply Copper Oxychloride 3g/L. Improve field drainage.",
        "prevention": "Crop rotation. Avoid excess nitrogen. Use certified seeds."
    },
    "rust": {
        "type": "disease",
        "severity": "Medium",
        "treatment": "Spray Propiconazole 1ml/L or Tebuconazole at 10-day intervals.",
        "prevention": "Use resistant varieties. Early planting to escape peak rust season."
    },
    "powdery_mildew": {
        "type": "disease",
        "severity": "Medium",
        "treatment": "Apply Sulphur 2g/L or Hexaconazole 1ml/L spray.",
        "prevention": "Adequate plant spacing for air circulation. Avoid high humidity."
    },
    "bacterial_blight": {
        "type": "disease",
        "severity": "High",
        "treatment": "Streptomycin + Copper Oxychloride spray. Remove infected material.",
        "prevention": "Use certified disease-free seeds. Avoid injuring plants."
    },
    "mosaic": {
        "type": "disease",
        "severity": "High",
        "treatment": "No cure — remove and destroy infected plants immediately.",
        "prevention": "Control aphid/whitefly vectors with Imidacloprid 0.3ml/L."
    },
    "weed": {
        "type": "weed",
        "severity": "Low",
        "treatment": "Manual removal or selective herbicide based on crop type.",
        "prevention": "Mulching, timely weeding, crop rotation."
    },
}

def get_treatment(class_name: str) -> dict:
    """Match class name to treatment info using partial string matching."""
    name_lower = class_name.lower().replace(" ", "_")
    for key, info in TREATMENTS.items():
        if key in name_lower:
            return info
    # Default fallback
    if "healthy" in name_lower:
        return TREATMENTS["healthy"]
    if "weed" in name_lower:
        return TREATMENTS["weed"]
    return {
        "type": "disease",
        "severity": "Unknown",
        "treatment": "Consult local agricultural extension officer for precise treatment.",
        "prevention": "Monitor regularly. Maintain good field hygiene."
    }

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Load image from bytes, resize to 224×224, normalize to [0,1].
    Matches training pipeline: rescale=1.0 with include_preprocessing=True
    (EfficientNetV2B0 handles its own internal normalization).
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE, Image.LANCZOS)
    img_array = np.array(img, dtype=np.float32)   # shape: (224, 224, 3), values [0,255]
    img_array = np.expand_dims(img_array, axis=0)  # shape: (1, 224, 224, 3)
    return img_array

# ── Routes ─────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def index():
    """Health check / info endpoint."""
    return jsonify({
        "status": "running",
        "model": "AgroVision v2 — EfficientNetV2B0",
        "classes": len(class_names),
        "input_size": f"{IMG_SIZE[0]}x{IMG_SIZE[1]}",
        "endpoints": {
            "POST /predict": "Upload image for crop/weed/disease detection",
            "GET /classes": "List all 38 class names"
        }
    })

@app.route("/classes", methods=["GET"])
def get_classes():
    """Return all class names the model can predict."""
    return jsonify({
        "total": len(class_names),
        "classes": class_names
    })

@app.route("/predict", methods=["POST"])
def predict():
    """
    Main prediction endpoint.

    Accepts:
        multipart/form-data with field 'image' (JPG / PNG / WEBP)

    Returns:
        {
          "success": true,
          "top_prediction": { "class": "...", "confidence": 0.97, "treatment": {...} },
          "top_3": [ { "class": "...", "confidence": 0.97 }, ... ],
          "image_size": "224x224"
        }
    """
    # ── Validate request ───────────────────────────────────────────────────
    if "image" not in request.files:
        return jsonify({"success": False, "error": "No image file found. Send as form-data with key 'image'."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"success": False, "error": "Empty filename."}), 400

    allowed_types = {"image/jpeg", "image/png", "image/webp", "image/jpg"}
    if file.content_type not in allowed_types:
        return jsonify({"success": False, "error": f"Unsupported file type: {file.content_type}. Use JPG/PNG/WEBP."}), 400

    try:
        # ── Read and preprocess ────────────────────────────────────────────
        image_bytes = file.read()
        img_array   = preprocess_image(image_bytes)

        # ── Run inference ──────────────────────────────────────────────────
        preds = model.predict(img_array, verbose=0)  # shape: (1, 38)
        preds = preds[0]                              # shape: (38,)

        # ── Get top K predictions ──────────────────────────────────────────
        top_k_indices = np.argsort(preds)[::-1][:TOP_K]

        top_3 = [
            {
                "class": class_names[i],
                "confidence": round(float(preds[i]), 4),
                "confidence_pct": f"{round(float(preds[i]) * 100, 1)}%"
            }
            for i in top_k_indices
        ]

        best = top_3[0]
        treatment = get_treatment(best["class"])

        return jsonify({
            "success": True,
            "top_prediction": {
                "class": best["class"],
                "confidence": best["confidence"],
                "confidence_pct": best["confidence_pct"],
                **treatment
            },
            "top_3": top_3,
            "image_size": f"{IMG_SIZE[0]}x{IMG_SIZE[1]}"
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ── Run ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
