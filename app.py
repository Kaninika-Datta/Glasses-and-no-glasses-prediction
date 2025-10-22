from flask import Flask, request, jsonify
import pickle
import numpy as np
from PIL import Image

app = Flask(__name__)

# Load your trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/")
def home():
    return "✅ Glasses Classifier API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    # Expecting an image file from the request
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']
    img = Image.open(file).resize((224, 224))  # same size as training
    img_array = np.expand_dims(np.array(img)/255.0, axis=0)  # normalize if needed

    # Make prediction
    prediction = model.predict(img_array)
    
    # Assuming output is categorical (0: glasses, 1: no-glasses)
    label = "glasses" if np.argmax(prediction) == 0 else "no-glasses"
    return jsonify({"prediction": label})

if __name__ == "__main__":
    app.run(debug=True)
