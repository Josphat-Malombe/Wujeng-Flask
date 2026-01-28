from flask import Flask, request, jsonify
from flask_cors import CORS
import model_def
from model_def import load_model, predict_from_bytes

app = Flask(__name__)
CORS(app)


load_model("resnet50_food101_mix_best.pth")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "Missing image"}), 400

    image_bytes = request.files["image"].read()
    return jsonify(predict_from_bytes(image_bytes))


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": model_def.MODEL is not None,
        "device": str(model_def.DEVICE)
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
