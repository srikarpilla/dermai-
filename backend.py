"""
backend.py — DermAI Integrated Server
Flask (Skin Prediction) + FastAPI (RAG Chatbot)
Production-ready for Render using Gunicorn
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.middleware.dispatcher import DispatcherMiddleware
import tensorflow as tf
import numpy as np
import json
from PIL import Image
import io
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

WEIGHTS_PATH     = os.path.join(BASE_DIR, "best_weights.weights.h5")
ARCH_PATH        = os.path.join(BASE_DIR, "model_architecture.json")
CLASS_NAMES_PATH = os.path.join(BASE_DIR, "class_names.json")
SYMPTOMS_PATH    = os.path.join(BASE_DIR, "symptoms.json")
MEDICINES_PATH   = os.path.join(BASE_DIR, "medicines.json")

IMG_SIZE = (224, 224)

# ─────────────────────────────────────────────
# Environment Variables
# ─────────────────────────────────────────────
SENDER_EMAIL    = os.environ.get("SENDER_EMAIL", "")
SENDER_PASSWORD = os.environ.get("SENDER_PASSWORD", "")

# ─────────────────────────────────────────────
# Flask App
# ─────────────────────────────────────────────
flask_app = Flask(__name__, static_folder=BASE_DIR, static_url_path='')
CORS(flask_app)

# ─────────────────────────────────────────────
# Load Model
# ─────────────────────────────────────────────
print("Loading model architecture...")
with open(ARCH_PATH, "r", encoding="utf-8") as f:
    model_json = f.read()

model = tf.keras.models.model_from_json(model_json)
model.load_weights(WEIGHTS_PATH)
print("Model loaded successfully.")

# Warm-up (prevents cold timeout)
try:
    print("Warming up model...")
    dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
    model.predict(dummy, verbose=0)
    print("Model ready.")
except:
    pass

# ─────────────────────────────────────────────
# Load Data
# ─────────────────────────────────────────────
with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
    class_names = json.load(f)

with open(SYMPTOMS_PATH, "r", encoding="utf-8") as f:
    DISEASE_SYMPTOMS = json.load(f)

with open(MEDICINES_PATH, "r", encoding="utf-8") as f:
    MEDICINES_DB = json.load(f)

# ─────────────────────────────────────────────
# Image Preprocessing (PIL)
# ─────────────────────────────────────────────
def preprocess_image(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    img = np.array(img, dtype=np.float32)
    img = (img / 127.5) - 1.0
    img = np.expand_dims(img, axis=0)
    return img


def get_medicine_info(disease):
    return MEDICINES_DB.get(disease, {})

# ─────────────────────────────────────────────
# Email Sender
# ─────────────────────────────────────────────
def send_email(recipient, disease, confidence):
    if not SENDER_EMAIL or not SENDER_PASSWORD:
        print("Email credentials not set.")
        return False

    try:
        msg = MIMEMultipart()
        msg["From"] = SENDER_EMAIL
        msg["To"] = recipient
        msg["Subject"] = f"DermAI Report — {disease}"

        body = f"""
DermAI Skin Report

Predicted Disease: {disease}
Confidence: {confidence}%

This is AI-generated.
Consult a licensed dermatologist.
"""
        msg.attach(MIMEText(body, "plain"))

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.send_message(msg)

        print("Email sent.")
        return True
    except Exception as e:
        print("Email failed:", e)
        return False

# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────
@flask_app.route("/")
def serve_index():
    return send_from_directory(BASE_DIR, "index.html")


@flask_app.route("/<path:filename>")
def serve_static(filename):
    return send_from_directory(BASE_DIR, filename)


@flask_app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]

        img_bytes = file.read()
        processed = preprocess_image(img_bytes)

        preds = model.predict(processed, verbose=0)[0]
        top_idx = int(np.argmax(preds))
        confidence = float(preds[top_idx] * 100)

        predicted_disease = class_names[top_idx]

        meds = get_medicine_info(predicted_disease)

        user_email = request.form.get("email", "")
        email_sent = False

        if user_email:
            email_sent = send_email(user_email, predicted_disease, f"{confidence:.2f}")

        return jsonify({
            "disease": predicted_disease,
            "confidence": f"{confidence:.2f}",
            "medicines": meds,
            "email_sent": email_sent
        })

    except Exception as e:
        print("Prediction error:", e)
        return jsonify({"error": "Prediction failed"}), 500


# ─────────────────────────────────────────────
# Mount FastAPI RAG Chatbot
# ─────────────────────────────────────────────
from derma_chat import app as fastapi_app
from a2wsgi import ASGIMiddleware

combined_app = DispatcherMiddleware(
    flask_app,
    {
        "/chat-ui": ASGIMiddleware(fastapi_app),
    }
)

# IMPORTANT:
# DO NOT add if __name__ == '__main__'
# Gunicorn will run this app.
