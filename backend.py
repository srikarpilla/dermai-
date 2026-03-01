from flask import Flask, request, jsonify, send_from_directory
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

app = Flask(__name__, static_folder='.', static_url_path='')

# ─────────────────────────────────────────────
# PATH SETUP (RENDER SAFE)
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ARCH_PATH        = os.path.join(BASE_DIR, "model_architecture.json")
WEIGHTS_PATH     = os.path.join(BASE_DIR, "best_weights.weights.h5")
CLASS_NAMES_PATH = os.path.join(BASE_DIR, "class_names.json")
SYMPTOMS_PATH    = os.path.join(BASE_DIR, "symptoms.json")
MEDICINES_PATH   = os.path.join(BASE_DIR, "medicines.json")

IMG_SIZE = (224, 224)
PORT = int(os.environ.get("PORT", 5000))  # Render uses dynamic port

# ─────────────────────────────────────────────
# EMAIL CONFIG (SET THESE IN RENDER ENV VARS)
# ─────────────────────────────────────────────
SENDER_EMAIL = os.environ.get("SENDER_EMAIL")
SENDER_PASSWORD = os.environ.get("SENDER_PASSWORD")

# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
print("Loading model architecture...")
with open(ARCH_PATH, "r", encoding="utf-8") as f:
    model_json = f.read()

model = tf.keras.models.model_from_json(model_json)

print("Loading trained weights...")
model.load_weights(WEIGHTS_PATH)

print("Model loaded successfully!")

# Warmup to prevent first-request lag
dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
model.predict(dummy)

# ─────────────────────────────────────────────
# LOAD JSON DATA
# ─────────────────────────────────────────────
with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
    class_names = json.load(f)

# Fix dict → list mapping safely
if isinstance(class_names, dict):
    class_names = [class_names[str(i)] for i in sorted(map(int, class_names.keys()))]

with open(SYMPTOMS_PATH, "r", encoding="utf-8") as f:
    DISEASE_SYMPTOMS = json.load(f)

with open(MEDICINES_PATH, "r", encoding="utf-8") as f:
    MEDICINES_DB = json.load(f)

print(f"{len(class_names)} classes loaded.")

# ─────────────────────────────────────────────
# IMAGE PREPROCESSING
# ─────────────────────────────────────────────
def preprocess_image(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    img = np.array(img, dtype=np.float32)

    # IMPORTANT: Use SAME normalization as training
    img = img / 255.0

    img = np.expand_dims(img, axis=0)
    return img

# ─────────────────────────────────────────────
# EMAIL FUNCTION
# ─────────────────────────────────────────────
def send_report_email(recipient_email, subject, body):
    if not SENDER_EMAIL or not SENDER_PASSWORD:
        print("Email credentials not set.")
        return False

    try:
        msg = MIMEMultipart()
        msg["From"] = SENDER_EMAIL
        msg["To"] = recipient_email
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.sendmail(SENDER_EMAIL, recipient_email, msg.as_string())

        print("Email sent successfully.")
        return True

    except Exception as e:
        print("Email error:", e)
        return False

# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────
@app.route("/")
def serve_index():
    return send_from_directory(".", "index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        # Parse user info
        user_info = {}
        if "user_info" in request.form:
            try:
                user_info = json.loads(request.form["user_info"])
            except:
                pass

        user_email = user_info.get("email", "")
        symptoms_text = user_info.get("symptoms", "")
        user_symptoms = [s.strip() for s in symptoms_text.replace(",", " ").split() if s.strip()]

        # Predict
        img_bytes = file.read()
        processed_img = preprocess_image(img_bytes)

        predictions = model.predict(processed_img)[0]

        top_idx = int(np.argmax(predictions))
        confidence = float(predictions[top_idx] * 100)
        predicted_disease = class_names[top_idx]

        print("Predicted:", predicted_disease, "| Confidence:", confidence)

        # Symptom matching
        known_symptoms = DISEASE_SYMPTOMS.get(predicted_disease, [])
        matching = [s for s in user_symptoms if any(s.lower() == k.lower() for k in known_symptoms)]
        missing  = [k for k in known_symptoms if not any(k.lower() == u.lower() for u in user_symptoms)]

        match_score = (
            f"{len(matching)} of {len(known_symptoms)} typical symptoms match"
            if known_symptoms else "No symptom data available"
        )

        meds = MEDICINES_DB.get(predicted_disease, {})

        # Send email
        email_sent = False
        if user_email:
            email_body = f"""
DermAI Report

Predicted Condition: {predicted_disease}
Confidence: {confidence:.2f}%
Symptom Alignment: {match_score}
"""
            email_sent = send_report_email(
                user_email,
                f"DermAI Report - {predicted_disease}",
                email_body
            )

        return jsonify({
            "disease": predicted_disease,
            "confidence": f"{confidence:.2f}",
            "match_score": match_score,
            "matching": matching,
            "missing": missing,
            "medicines": meds,
            "email_sent": email_sent
        })

    except Exception as e:
        print("Prediction error:", e)
        return jsonify({"error": "Prediction failed"}), 500

# ─────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Starting DermAI server on port {PORT}")
    app.run(host="0.0.0.0", port=PORT, debug=False)
