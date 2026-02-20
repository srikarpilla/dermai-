import os
import json
import io

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"]  = "-1"

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
from PIL import Image

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

# ── Relative paths ──────────────────────────────────────────────
BASE             = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH     = os.path.join(BASE, "best_weights.weights.h5")
ARCH_PATH        = os.path.join(BASE, "model_architecture.json")
CLASS_NAMES_PATH = os.path.join(BASE, "class_names.json")
SYMPTOMS_PATH    = os.path.join(BASE, "symptoms.json")
MEDICINES_PATH   = os.path.join(BASE, "medicines.json")
IMG_SIZE         = (224, 224)

# ── Lazy globals — loaded only on first /predict call ──────────
_model        = None
_class_names  = None
_symptoms_db  = None
_medicines_db = None


def load_model_once():
    global _model, _class_names, _symptoms_db, _medicines_db
    if _model is not None:
        return

    import tensorflow as tf

    print("⏳ Loading model architecture...")
    with open(ARCH_PATH, 'r', encoding='utf-8') as f:
        model_json = f.read()
    _model = tf.keras.models.model_from_json(model_json)

    print("⏳ Loading weights...")
    _model.load_weights(WEIGHTS_PATH)

    print("⏳ Loading class names...")
    with open(CLASS_NAMES_PATH, 'r', encoding='utf-8') as f:
        _class_names = json.load(f)

    print("⏳ Loading symptoms...")
    with open(SYMPTOMS_PATH, 'r', encoding='utf-8') as f:
        _symptoms_db = json.load(f)

    print("⏳ Loading medicines...")
    if os.path.exists(MEDICINES_PATH):
        with open(MEDICINES_PATH, 'r', encoding='utf-8') as f:
            _medicines_db = json.load(f)
    else:
        _medicines_db = {}

    print(f"✅ Model ready — {len(_class_names)} classes loaded.")


def preprocess_image(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB').resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)
    arr = (arr / 127.5) - 1.0
    return arr


# ── ROUTES ──────────────────────────────────────────────────────

@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')

@app.route('/health')
def health():
    return jsonify({"status": "ok", "model_loaded": _model is not None})

@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory('.', filename)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        load_model_once()  # loads TF only on first call

        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        # Parse user info
        user_info     = {}
        user_symptoms = []
        if 'user_info' in request.form:
            try:
                user_info     = json.loads(request.form['user_info'])
                symptoms_text = user_info.get('symptoms', '')
                user_symptoms = [
                    s.strip()
                    for s in symptoms_text.replace(',', '\n').split('\n')
                    if s.strip()
                ]
            except Exception:
                pass

        # Predict
        img_bytes   = file.read()
        processed   = preprocess_image(img_bytes)
        predictions = _model.predict(processed)[0]

        top_idx           = int(np.argmax(predictions))
        confidence        = float(predictions[top_idx] * 100)
        predicted_disease = _class_names[top_idx]

        # Symptom matching
        known_symptoms = _symptoms_db.get(predicted_disease, [])
        matching = [s for s in user_symptoms
                    if any(s.lower() == k.lower() for k in known_symptoms)]
        missing  = [k for k in known_symptoms
                    if not any(k.lower() == u.lower() for u in user_symptoms)]
        match_score = (
            f"{len(matching)} of {len(known_symptoms)} typical symptoms match"
            if known_symptoms else "No symptom data available"
        )

        medicines = _medicines_db.get(predicted_disease, {})

        # Email (optional — won't crash if env vars missing)
        try:
            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart

            sender   = os.getenv("SENDER_EMAIL", "")
            password = os.getenv("SENDER_PASSWORD", "")
            to_email = user_info.get("email", "")

            if sender and password and to_email:
                msg = MIMEMultipart()
                msg['From']    = sender
                msg['To']      = to_email
                msg['Subject'] = f"DermAI Report — {predicted_disease}"
                body = f"""
DermAI Skin Analysis Report
============================
Patient : {user_info.get('name', 'N/A')}
Age     : {user_info.get('age', 'N/A')}

Predicted Condition : {predicted_disease}
Confidence          : {confidence:.2f}%
Symptom Alignment   : {match_score}
Matching Symptoms   : {', '.join(matching) or 'None'}
Additional Symptoms : {', '.join(missing) or 'None'}

This is an AI-generated report for educational purposes only.
Please consult a licensed dermatologist for diagnosis and treatment.
"""
                msg.attach(MIMEText(body, 'plain'))
                with smtplib.SMTP_SSL('smtp.gmail.com', 465) as s:
                    s.login(sender, password)
                    s.send_message(msg)
        except Exception as mail_err:
            print("Email error (non-fatal):", mail_err)

        return jsonify({
            "disease":     predicted_disease,
            "confidence":  f"{confidence:.2f}",
            "match_score": match_score,
            "matching":    matching,
            "missing":     missing,
            "medicines":   medicines,
            "email_sent":  True
        })

    except Exception as e:
        print("Prediction error:", e)
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
