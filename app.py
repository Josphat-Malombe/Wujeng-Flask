from flask import Flask, request, jsonify
from flask_cors import CORS
import model_def
from model_def import load_model, predict_from_bytes
import logging,os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)
"""
MODEL_PATH = os.getenv("MODEL_PATH","resnet50_food101_mix_best.pth")

try:
    load_model(MODEL_PATH)
    logger.info(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    logger.error(f"Model load failed {str(e)}")
    raise
"""

load_model("resnet50_food101_mix_best.pth")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "Missing image"}), 400
    file=request.files["image"]
    if file.filename == "":
        return jsonify({"error": "no file selected"}), 400
    
    try:
        image_bytes = file.read()
        result = predict_from_bytes(image_bytes)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Prediction failed {str(e)}")
        return jsonify({"error":"Internal server error during predict"})

   


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": model_def.MODEL is not None,
        "device": str(model_def.DEVICE)
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
