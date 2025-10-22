from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask_cors import CORS
import numpy as np
from PIL import Image
import io
import base64

app = Flask(__name__)
CORS(app)  # Allow frontend access

# Load model once
model = load_model("model.pkl")

@app.route('/')
def home():
    return "✅ Glasses Classifier API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        img_data = data.get("image")

        if not img_data:
            return jsonify({"error": "No image provided"}), 400

        # Decode base64 image
        img_bytes = base64.b64decode(img_data.split(",")[1])
        img = Image.open(io.BytesIO(img_bytes)).resize((128, 128))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)[0][0]
        result = "Wearing Glasses 😎" if prediction > 0.5 else "No Glasses 🙈"

        return jsonify({"result": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
