from flask import Flask, request, jsonify, send_from_directory
import tensorflow as tf
import numpy as np
import json
from PIL import Image
import io
import os
import smtplib
import threading
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# ─────────────────────────────────────────────
# CREATE FLASK APP
# ─────────────────────────────────────────────
app = Flask(__name__, static_folder='.', static_url_path='')
print("Flask app initialized")

# ─────────────────────────────────────────────
# PATH SETUP
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ARCH_PATH        = os.path.join(BASE_DIR, "model_architecture.json")
WEIGHTS_PATH     = os.path.join(BASE_DIR, "best_weights.weights.h5")
CLASS_NAMES_PATH = os.path.join(BASE_DIR, "class_names.json")
SYMPTOMS_PATH    = os.path.join(BASE_DIR, "symptoms.json")
MEDICINES_PATH   = os.path.join(BASE_DIR, "medicines.json")

IMG_SIZE = (224, 224)

# ─────────────────────────────────────────────
# EMAIL
# ─────────────────────────────────────────────
SENDER_EMAIL    = os.environ.get("SENDER_EMAIL")
SENDER_PASSWORD = os.environ.get("SENDER_PASSWORD")

# ─────────────────────────────────────────────
# LOAD MODEL — with explicit graph/session fix
# ─────────────────────────────────────────────
print("Loading model architecture...")

with open(ARCH_PATH, "r", encoding="utf-8") as f:
    model_json = f.read()

model = tf.keras.models.model_from_json(model_json)

print("Loading best weights...")
model.load_weights(WEIGHTS_PATH)

# ✅ FIX 1: Build the model's predict function ahead of time
#    This prevents the "graph is finalized" / silent hang on first real request
model.predict(np.zeros((1, 224, 224, 3), dtype=np.float32))
print("MODEL LOADED AND WARMED UP SUCCESSFULLY")

# ✅ FIX 2: Global threading lock — prevents TF race conditions
#    under multi-threaded or multi-request Flask/Gunicorn scenarios
predict_lock = threading.Lock()

# ─────────────────────────────────────────────
# LOAD JSON FILES
# ─────────────────────────────────────────────
with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
    class_names = json.load(f)

if isinstance(class_names, dict):
    class_names = [class_names[str(i)] for i in sorted(map(int, class_names.keys()))]

# ✅ FIX 3: Graceful fallback if symptoms.json or medicines.json missing
try:
    with open(SYMPTOMS_PATH, "r", encoding="utf-8") as f:
        DISEASE_SYMPTOMS = json.load(f)
except FileNotFoundError:
    print("WARNING: symptoms.json not found — symptom matching disabled.")
    DISEASE_SYMPTOMS = {}

try:
    with open(MEDICINES_PATH, "r", encoding="utf-8") as f:
        MEDICINES_DB = json.load(f)
except FileNotFoundError:
    print("WARNING: medicines.json not found — medicine recommendations disabled.")
    MEDICINES_DB = {}

print(f"{len(class_names)} classes ready.")

# ─────────────────────────────────────────────
# IMAGE PREPROCESSING
# ─────────────────────────────────────────────
def preprocess_image(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    img = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(img, axis=0)

# ─────────────────────────────────────────────
# EMAIL FUNCTION
# ─────────────────────────────────────────────
def send_report_email(recipient_email, subject, body):
    if not SENDER_EMAIL or not SENDER_PASSWORD:
        print("Email credentials not configured.")
        return False
    try:
        msg = MIMEMultipart()
        msg["From"]    = SENDER_EMAIL
        msg["To"]      = recipient_email
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.sendmail(SENDER_EMAIL, recipient_email, msg.as_string())
        print("Email sent.")
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


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "classes": len(class_names),
        "symptoms_loaded": bool(DISEASE_SYMPTOMS),
        "medicines_loaded": bool(MEDICINES_DB),
    })


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # ── Validate file ──────────────────────────────────────
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        # ── Parse user info (optional — won't block prediction) ──
        user_info = {}
        if "user_info" in request.form:
            try:
                user_info = json.loads(request.form["user_info"])
            except Exception:
                pass

        user_email    = user_info.get("email", "")
        symptoms_text = user_info.get("symptoms", "")

        # ── Preprocess ─────────────────────────────────────────
        img_bytes     = file.read()
        processed_img = preprocess_image(img_bytes)

        # ── ✅ FIX 2 applied: Predict inside the lock ──────────
        with predict_lock:
            predictions = model.predict(processed_img, verbose=0)[0]

        print("Raw predictions (top-3):", sorted(enumerate(predictions), key=lambda x: -x[1])[:3])

        top_idx           = int(np.argmax(predictions))
        confidence        = float(predictions[top_idx] * 100)
        predicted_disease = class_names[top_idx]

        print(f"Predicted: {predicted_disease} | Confidence: {confidence:.2f}%")

        # ── Symptom matching ───────────────────────────────────
        user_symptoms  = [s.strip() for s in symptoms_text.replace(",", " ").split() if s.strip()]
        known_symptoms = DISEASE_SYMPTOMS.get(predicted_disease, [])

        matching = [s for s in user_symptoms if any(s.lower() == k.lower() for k in known_symptoms)]
        missing  = [k for k in known_symptoms if not any(k.lower() == u.lower() for u in user_symptoms)]

        match_score = (
            f"{len(matching)} of {len(known_symptoms)} typical symptoms match"
            if known_symptoms else "No symptom data available"
        )

        meds = MEDICINES_DB.get(predicted_disease, {})

        # ── Email report ───────────────────────────────────────
        email_sent = False
        if user_email:
            email_body = (
                f"DermAI Report\n\n"
                f"Predicted Condition: {predicted_disease}\n"
                f"Confidence: {confidence:.2f}%\n"
                f"Symptom Alignment: {match_score}\n"
            )
            email_sent = send_report_email(
                user_email,
                f"DermAI Report - {predicted_disease}",
                email_body
            )

        return jsonify({
            "disease":    predicted_disease,
            "confidence": f"{confidence:.2f}",
            "match_score": match_score,
            "matching":   matching,
            "missing":    missing,
            "medicines":  meds,
            "email_sent": email_sent,
        })

    except Exception as e:
        # ✅ FIX 3: Log the REAL error so you can debug it on Render
        import traceback
        print("=== PREDICTION ERROR ===")
        traceback.print_exc()
        print("========================")
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500
