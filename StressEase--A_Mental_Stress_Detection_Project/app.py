import os
import base64
import json
import sqlite3
import logging
from datetime import datetime, UTC
from flask import Flask, render_template, request, jsonify, redirect, url_for, abort, session
from pymongo import MongoClient
from werkzeug.security import check_password_hash, generate_password_hash
import joblib
import numpy as np
import pandas as pd

# ---------------- Config & Logging ----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("StressEaseAI")

app = Flask(__name__, template_folder="templates")
app.secret_key = os.getenv("SECRET_KEY", "stressease-dev-secret")

# ---------------- Optional MongoDB Setup ----------------
# MongoDB is optional in this project. If MongoDB is not running, the app still works
# because important records are also saved in SQLite through save_sqlite_record().
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=1000)
db = mongo_client.get_database("StressPredictionDB")
conversation_collection = db.get_collection("Conversations")
user_inputs_collection = db.get_collection("UserInputs")
SQLITE_DB_PATH = os.getenv("SQLITE_DB_PATH",  "stressease.db")


def safe_insert(collection, document):
    # This helper prevents MongoDB connection errors from stopping the Flask app.
    try:
        collection.insert_one(document)
    except Exception as e:
        logger.warning("Skipping MongoDB write. Error: %s", e)


def init_sqlite():
    # SQLite creates a local file named StressEase.db.
    # users table: stores signup/login data.
    # monitoring_records table: stores stress, mood, questionnaire, face, and chat logs.
    with sqlite3.connect(SQLITE_DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS monitoring_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                record_type TEXT NOT NULL,
                payload TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )


def save_sqlite_record(record_type, payload):
    # Stores different module outputs in one table.
    # record_type tells which module generated the data.
    # payload is saved as JSON text so flexible data can be stored easily.
    try:
        with sqlite3.connect(SQLITE_DB_PATH) as conn:
            conn.execute(
                "INSERT INTO monitoring_records (record_type, payload, created_at) VALUES (?, ?, ?)",
                (record_type, json.dumps(payload), datetime.now(UTC).isoformat()),
            )
    except Exception as e:
        logger.warning("Skipping SQLite write. Error: %s", e)


init_sqlite()  # Ensure SQLite tables exist when the app starts.

# ---------------- ML Model ----------------
MODEL_PATH = os.getenv("MODEL_PATH", "model_rf.pkl")
# These are the exact input features expected by the trained Random Forest model.
MODEL_FEATURES = [
    "anxiety_level",
    "self_esteem",
    "mental_health_history",
    "depression",
    "headache",
    "blood_pressure",
    "sleep_quality",
    "breathing_problem",
    "noise_level",
    "living_conditions",
    "safety",
    "basic_needs",
    "academic_performance",
    "study_load",
    "teacher_student_relationship",
    "future_career_concerns",
    "social_support",
    "peer_pressure",
    "extracurricular_activities",
    "bullying",
]

try:
    # model_rf.pkl contains the trained stress prediction model.
    model = joblib.load(MODEL_PATH)
    logger.info("ML model loaded from %s", MODEL_PATH)
except Exception as e:
    model = None
    logger.warning("Could not load ML model (%s). Continuing without it. Error: %s", MODEL_PATH, e)

# ---------------- OpenRouter / GenAI Setup ----------------
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "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")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")  # kept for backward compatibility
DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
DEFAULT_OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-3-ultra-550b-a55b:free")

# FER2013 CNN is kept as a future enhancement. If a trained .h5 file is added,
# this hook can load it. Otherwise the project uses OpenCV-based face analysis.
FER_CNN_MODEL_PATH = os.getenv("FER_CNN_MODEL_PATH", "fer2013_cnn_model.h5")
fer_cnn_model = None
fer_cnn_status = "FER2013 CNN model file not found; OpenCV fallback active."

if os.path.exists(FER_CNN_MODEL_PATH):
    try:
        from tensorflow.keras.models import load_model

        fer_cnn_model = load_model(FER_CNN_MODEL_PATH)
        fer_cnn_status = f"FER2013 CNN model loaded from {FER_CNN_MODEL_PATH}."
    except Exception as e:
        fer_cnn_status = f"FER2013 CNN model could not load; OpenCV fallback active. Error: {e}"
        logger.warning(fer_cnn_status)

try:
    # OpenCV is used for webcam-frame based facial expression signal analysis.
    import cv2
except Exception as e:
    cv2 = None
    logger.warning("OpenCV unavailable. Facial expression detection disabled. Error: %s", e)


def analyze_mood_text(text):
    # Simple rule-based mood analysis.
    # Positive words increase score and stress-related words decrease score.
    text = (text or "").lower()
    positive_words = {
        "calm", "happy", "good", "great", "hopeful", "focused", "relaxed",
        "better", "confident", "fine", "okay", "excited", "motivated"
    }
    negative_words = {
        "stress", "stressed", "anxious", "anxiety", "sad", "tired", "angry",
        "panic", "worried", "overwhelmed", "depressed", "lonely", "bad",
        "headache", "fear", "pressure", "cry", "hopeless"
    }
    score = sum(1 for word in positive_words if word in text)
    score -= sum(1 for word in negative_words if word in text)

    if score <= -2:
        mood = "Distressed"
        risk = "High"
    elif score == -1:
        mood = "Low"
        risk = "Moderate"
    elif score == 0:
        mood = "Neutral"
        risk = "Mild"
    else:
        mood = "Positive"
        risk = "Low"

    return {
        "mood": mood,
        "risk": risk,
        "sentiment_score": score,
        "recommendation": recommendation_for_risk(risk),
    }


def recommendation_for_risk(risk):
    # Converts risk level into a short wellness recommendation for the user.
    if risk == "High":
        return "Pause, use slow breathing for 2 minutes, and reach out to a trusted person or counselor if this feeling continues."
    if risk == "Moderate":
        return "Take a short break, reduce immediate workload, hydrate, and log your mood again later today."
    if risk == "Mild":
        return "Keep monitoring your mood and follow one small self-care action such as walking or journaling."
    return "Maintain your routine, sleep schedule, and regular check-ins."


def severity_from_score(score, thresholds):
    # Shared helper to convert numeric questionnaire scores into severity labels.
    for label, upper_bound in thresholds:
        if score <= upper_bound:
            return label
    return thresholds[-1][0]


def score_pss(answers):
    # PSS-10 scoring: measures perceived stress in daily life.
    # Some PSS questions are reverse scored as per the standard scoring method.
    values = [int(value) for value in answers]
    reverse_indices = {3, 4, 6, 7}
    total = sum((4 - value) if index in reverse_indices else value for index, value in enumerate(values))
    if total <= 13:
        level = "Low stress"
        risk = "Low"
    elif total <= 26:
        level = "Moderate stress"
        risk = "Moderate"
    else:
        level = "High perceived stress"
        risk = "High"
    return {
        "test": "PSS-10",
        "score": total,
        "max_score": 40,
        "level": level,
        "risk": risk,
        "recommendation": recommendation_for_risk(risk),
    }


def score_dass21(answers):
    # DASS-21 scoring: separate scores for Depression, Anxiety, and Stress.
    # Each subscale has 7 questions; final scores are multiplied by 2.
    values = [int(value) for value in answers]
    depression_indices = [2, 4, 9, 12, 15, 16, 20]
    anxiety_indices = [1, 3, 6, 8, 14, 18, 19]
    stress_indices = [0, 5, 7, 10, 11, 13, 17]

    depression = sum(values[index] for index in depression_indices) * 2
    anxiety = sum(values[index] for index in anxiety_indices) * 2
    stress = sum(values[index] for index in stress_indices) * 2

    result = {
        "test": "DASS-21",
        "depression_score": depression,
        "anxiety_score": anxiety,
        "stress_score": stress,
        "depression_level": severity_from_score(
            depression,
            [("Normal", 9), ("Mild", 13), ("Moderate", 20), ("Severe", 27), ("Extremely Severe", 999)],
        ),
        "anxiety_level": severity_from_score(
            anxiety,
            [("Normal", 7), ("Mild", 9), ("Moderate", 14), ("Severe", 19), ("Extremely Severe", 999)],
        ),
        "stress_level": severity_from_score(
            stress,
            [("Normal", 14), ("Mild", 18), ("Moderate", 25), ("Severe", 33), ("Extremely Severe", 999)],
        ),
    }
    risk = "High" if stress >= 26 or depression >= 21 or anxiety >= 15 else "Moderate" if stress >= 15 or depression >= 10 or anxiety >= 8 else "Low"
    result["risk"] = risk
    result["recommendation"] = recommendation_for_risk(risk)
    return result


def stress_label(prediction):
    # Converts ML model numeric output into a user-friendly label.
    labels = {
        0: "Low stress",
        1: "Moderate stress",
        2: "High stress",
    }
    try:
        return labels.get(int(prediction), f"Stress level {prediction}")
    except Exception:
        return "Stress level unavailable"


def build_chat_prompt(user_msg, history):
    # Builds a safe mental-wellness prompt for Gemini using recent chat history.
    recent_history = history[-8:] if isinstance(history, list) else []
    transcript = []
    for item in recent_history:
        role = item.get("role", "user")
        text = str(item.get("text", "")).strip()
        if text:
            transcript.append(f"{role.title()}: {text}")

    return (
        "You are StressEase AI, a ChatGPT-style assistant inside an AI-Powered Stress and Mood "
        "Detection System for Mental Health Monitoring.\n"
        "Rules:\n"
        "- Answer naturally and conversationally.\n"
        "- Remember and use the recent conversation context.\n"
        "- Be helpful for study stress, mood tracking, project explanation, and wellness guidance.\n"
        "- Do not claim to be a doctor or provide diagnosis.\n"
        "- If the user mentions self-harm, emergency, or danger, advise contacting local emergency help or a trusted person immediately.\n"
        "- Keep answers clear, practical, and not too long unless the user asks for detail.\n\n"
        "Recent conversation:\n"
        f"{chr(10).join(transcript) if transcript else 'No previous messages.'}\n\n"
        f"User: {user_msg}\n"
        "Assistant:"
    )


def analyze_face_frame(image_data):
    # Frontend sends webcam image as base64. Backend decodes it and analyzes it using OpenCV.
    if cv2 is None:
        return {"error": "OpenCV is not installed in this environment."}, 503

    # Remove browser data URL prefix if present.
    if "," in image_data:
        image_data = image_data.split(",", 1)[1]

    image_bytes = base64.b64decode(image_data)
    buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if frame is None:
        return {"error": "Invalid camera frame."}, 400

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = []
    smiles = []
    if hasattr(cv2, "CascadeClassifier"):
        # Preferred path: Haar cascade face and smile detection.
        face_detector = cv2.CascadeClassifier(
            os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        )
        smile_detector = cv2.CascadeClassifier(
            os.path.join(cv2.data.haarcascades, "haarcascade_smile.xml")
        )
        faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
    else:
        # Fallback path for OpenCV builds without CascadeClassifier.
        # It uses skin-tone masking and a central face-region estimate.
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        skin_mask = cv2.inRange(hsv, np.array([0, 20, 45]), np.array([30, 255, 255]))
        skin_ratio = float(np.count_nonzero(skin_mask)) / float(skin_mask.size)
        if skin_ratio > 0.03:
            height, width = gray.shape
            box_width = int(width * 0.38)
            box_height = int(height * 0.48)
            x = int((width - box_width) / 2)
            y = int(height * 0.18)
            faces = np.array([[x, y, box_width, box_height]])

    if len(faces) == 0:
        # Return a friendly response instead of crashing when no face is detected.
        return {
            "face_detected": False,
            "emotion": "No face detected",
            "stress_signal": "Unknown",
            "confidence": 0,
            "face_count": 0,
            "recommendation": "Keep your face centered in good lighting and try again.",
        }, 200

    x, y, w, h = max(faces, key=lambda face: face[2] * face[3])
    face_roi = gray[y:y + h, x:x + w]
    if hasattr(cv2, "CascadeClassifier"):
        smiles = smile_detector.detectMultiScale(face_roi, scaleFactor=1.7, minNeighbors=18, minSize=(25, 25))
    brightness = float(np.mean(face_roi))
    contrast = float(np.std(face_roi))

    lower_face = face_roi[int(h * 0.55):, :]
    mouth_contrast = float(np.std(lower_face)) if lower_face.size else contrast

    # Basic emotion signal estimation from smile count, brightness, and contrast.
    if len(smiles) > 0 or (brightness > 95 and mouth_contrast > 42):
        emotion = "Happy"
        stress_signal = "Low"
        confidence = min(94, 70 + len(smiles) * 8)
    elif brightness < 70 or contrast > 68:
        emotion = "Tense"
        stress_signal = "Moderate"
        confidence = 62
    else:
        emotion = "Neutral"
        stress_signal = "Mild"
        confidence = 58

    risk = "Low" if stress_signal == "Low" else "Moderate" if stress_signal == "Moderate" else "Mild"
    return {
        "face_detected": True,
        "emotion": emotion,
        "stress_signal": stress_signal,
        "confidence": confidence,
        "model_status": fer_cnn_status,
        "face_count": int(len(faces)),
        "box": {"x": int(x), "y": int(y), "width": int(w), "height": int(h)},
        "recommendation": recommendation_for_risk(risk),
    }, 200

# ---------------- Routes ----------------
@app.route("/")
def root():
    # First page of the app opens signup/login flow.
    return redirect(url_for("signup"))

@app.route("/<page_name>.html")
def template_page(page_name):
    # Allows direct opening of template pages like /mood-tracker.html safely.
    if "/" in page_name or "\\" in page_name or page_name.startswith("."):
        abort(404)
    template = f"{page_name}.html"
    template_path = os.path.join(app.template_folder, template)
    if os.path.isfile(template_path):
        return render_template(template)
    abort(404)

@app.route("/signup", methods=["GET", "POST"])
def signup():
    # New user registration. Password is hashed before storing in SQLite.
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        if not name or not email or not password:
            return render_template("signup.html", error="Please fill all fields.")

        try:
            with sqlite3.connect(SQLITE_DB_PATH) as conn:
                conn.execute(
                    "INSERT INTO users (name, email, password_hash, created_at) VALUES (?, ?, ?, ?)",
                    (name, email, generate_password_hash(password), datetime.now(UTC).isoformat()),
                )
            session["user_email"] = email
            session["user_name"] = name
            return redirect(url_for("home"))
        except sqlite3.IntegrityError:
            return render_template("signup.html", error="Account already exists. Please login.")
    return render_template("signup.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    # Login checks email in SQLite and verifies hashed password.
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        with sqlite3.connect(SQLITE_DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            user = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        if user and check_password_hash(user["password_hash"], password):
            session["user_email"] = user["email"]
            session["user_name"] = user["name"]
            return redirect(url_for("home"))
        return render_template("login.html", error="Invalid email or password.")
    return render_template("login.html")

@app.route("/logout")
def logout():
    # Clears Flask session so the user is logged out.
    session.clear()
    return redirect(url_for("login"))

@app.route("/home")
def home():
    return render_template("index.html")

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

@app.route("/questionnaires")
def questionnaires():
    return render_template("questionnaires.html")

@app.route("/mood-calendar")
def mood_calendar():
    return render_template("calendar-reminders.html")

@app.route("/face-detection")
def face_detection():
    return render_template("real-time-monitor.html")

@app.route("/forum")
def forum():
    return render_template("forum.html")

@app.route("/journal", methods=["GET", "POST"])
def journal():
    # Optional journal page: saves free-text journal entries if MongoDB is available.
    if request.method == "POST":
        entry = request.form.get("journal_entry")
        if entry:
            safe_insert(user_inputs_collection, {"journal_entry": entry})
    return render_template("journal.html")

@app.route("/assessment", methods=["GET", "POST"])
def assessment():
    # Legacy assessment route kept for compatibility with older form pages.
    if request.method == "POST":
        answers = {k: request.form[k] for k in request.form}
        safe_insert(user_inputs_collection, answers)
        return redirect(url_for("predict"))
    return render_template("predict.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    # Form-based stress prediction route.
    # Converts slider values into model features and returns a result page.
    if request.method == "POST":
        try:
            data = {feature: float(request.form[feature]) for feature in MODEL_FEATURES}
            input_df = pd.DataFrame([data], columns=MODEL_FEATURES)
            prediction = model.predict(input_df)[0] if model else None
            if prediction is not None:
                safe_insert(user_inputs_collection, {**data, "predicted_stress_level": float(prediction)})
                save_sqlite_record("stress_prediction", {**data, "predicted_stress_level": float(prediction)})
            return render_template(
                "result.html",
                prediction=prediction,
                label=stress_label(prediction),
                recommendation=recommendation_for_risk("High" if int(prediction) == 2 else "Moderate" if int(prediction) == 1 else "Low") if prediction is not None else None,
            )
        except Exception as e:
            logger.exception("Prediction error: %s", e)
            return render_template("result.html", error="Error occurred during prediction.")
    return render_template("predict.html")

@app.route("/api/predict-stress", methods=["POST"])
def api_predict_stress():
    # JSON API used by the stress detector page.
    # Frontend sends slider values; backend returns prediction, label, and recommendation.
    try:
        payload = request.get_json(force=True) or {}
        data = {feature: float(payload.get(feature, 0)) for feature in MODEL_FEATURES}
        input_df = pd.DataFrame([data], columns=MODEL_FEATURES)
        prediction = model.predict(input_df)[0] if model else None
        result = {
            "prediction": int(prediction) if prediction is not None else None,
            "label": stress_label(prediction),
            "features": data,
        }
        risk = "High" if result["prediction"] == 2 else "Moderate" if result["prediction"] == 1 else "Low"
        result["recommendation"] = recommendation_for_risk(risk)
        safe_insert(user_inputs_collection, {**data, "predicted_stress_level": result["prediction"]})
        save_sqlite_record("stress_prediction", result)
        return jsonify(result)
    except Exception as e:
        logger.exception("API prediction error: %s", e)
        return jsonify({"error": "Prediction failed. Please check input values."}), 400

@app.route("/api/analyze-mood", methods=["POST"])
def api_analyze_mood():
    # JSON API for mood tracker text analysis.
    payload = request.get_json(force=True) or {}
    text = payload.get("text", "")
    result = analyze_mood_text(text)
    safe_insert(user_inputs_collection, {"mood_text": text, **result})
    save_sqlite_record("mood_analysis", {"mood_text": text, **result})
    return jsonify(result)

@app.route("/api/submit-questionnaire", methods=["POST"])
def api_submit_questionnaire():
    # JSON API for DASS-21 and PSS-10 questionnaire scoring.
    try:
        payload = request.get_json(force=True) or {}
        test_type = payload.get("test_type")
        answers = payload.get("answers", [])

        if test_type == "pss":
            if len(answers) != 10:
                return jsonify({"error": "PSS-10 requires 10 answers."}), 400
            result = score_pss(answers)
        elif test_type == "dass":
            if len(answers) != 21:
                return jsonify({"error": "DASS-21 requires 21 answers."}), 400
            result = score_dass21(answers)
        else:
            return jsonify({"error": "Unknown questionnaire type."}), 400

        result["answers"] = answers
        safe_insert(user_inputs_collection, {"questionnaire": result})
        save_sqlite_record("questionnaire", result)
        return jsonify(result)
    except Exception as e:
        logger.exception("Questionnaire scoring error: %s", e)
        return jsonify({"error": "Unable to score questionnaire."}), 400

@app.route("/api/analyze-face", methods=["POST"])
def api_analyze_face():
    # JSON API for webcam facial-expression analysis.
    payload = request.get_json(force=True) or {}
    image_data = payload.get("image", "")
    if not image_data:
        return jsonify({"error": "Missing camera frame."}), 400
    result, status = analyze_face_frame(image_data)
    if status == 200:
        safe_insert(user_inputs_collection, {"face_expression": result})
        save_sqlite_record("face_expression", result)
    return jsonify(result), status

@app.route("/api/fer-model-status")
def api_fer_model_status():
    # Shows whether future FER2013 CNN model file is loaded or OpenCV fallback is active.
    return jsonify({
        "cnn_model_loaded": fer_cnn_model is not None,
        "model_path": FER_CNN_MODEL_PATH,
        "status": fer_cnn_status,
    })

@app.route("/recommendations")
def recommendations():
    return render_template("recommendations.html")

@app.route("/chat")
def chat_page():
    return render_template("chat.html")

@app.route("/chatbot", methods=["POST"])
def chatbot():
    # Chatbot endpoint. Uses OpenRouter API first, then Gemini, then local fallback.
    try:
        data = request.get_json(force=True)
        user_msg = (data or {}).get("message", "").strip()
        history = (data or {}).get("history", [])
        if not user_msg:
            return jsonify({"response": "Please send a message."}), 400

        safe_insert(conversation_collection, {"role": "user", "text": user_msg})
        save_sqlite_record("chat_user", {"text": user_msg})

        bot_reply = None

        # --- OpenRouter path ---
        if OPENROUTER_API_KEY:
            try:
                from openai import OpenAI as OpenRouterClient
                client = OpenRouterClient(
                    api_key=OPENROUTER_API_KEY,
                    base_url="https://openrouter.ai/api/v1",
                )
                completion = client.chat.completions.create(
                    model=DEFAULT_OPENROUTER_MODEL,
                    messages=[{"role": "user", "content": build_chat_prompt(user_msg, history)}],
                    max_tokens=650,
                    temperature=0.7,
                )
                bot_reply = completion.choices[0].message.content.strip()
            except Exception as or_error:
                logger.warning("OpenRouter unavailable. Error: %s", or_error)

        # --- Gemini fallback path ---
        if not bot_reply and GEMINI_API_KEY:
            try:
                from google import genai
                from google.genai import types
                genai_client = genai.Client(api_key=GEMINI_API_KEY)
                response = genai_client.models.generate_content(
                    model=DEFAULT_GEMINI_MODEL,
                    contents=build_chat_prompt(user_msg, history),
                    config=types.GenerateContentConfig(
                        max_output_tokens=650,
                        temperature=0.7,
                    ),
                )
                bot_reply = getattr(response, "text", str(response)).strip()
            except Exception as ai_error:
                logger.warning("Gemini unavailable. Error: %s", ai_error)

        # --- Local fallback (last resort) ---
        if not bot_reply:
            logger.warning("All AI providers unavailable. Using local mood fallback.")
            mood = analyze_mood_text(user_msg)
            bot_reply = (
                f"I detected a {mood['mood'].lower()} mood signal with {mood['risk'].lower()} risk. "
                f"{mood['recommendation']}"
            )

        safe_insert(conversation_collection, {"role": "bot", "text": bot_reply})
        save_sqlite_record("chat_bot", {"text": bot_reply})

        return jsonify({"response": bot_reply, "stress_level": "neutral"})

    except Exception as ex:
        logger.exception("Error in /chatbot: %s", ex)
        return jsonify({"response": "Something went wrong. Try again later.", "error": str(ex)}), 500

# ---------------- Run ----------------
if __name__ == "__main__":
    # PORT can be changed from terminal, for example: PORT=5001 python app.py
    debug_mode = os.getenv("FLASK_DEBUG", "0") == "1"
    port = int(os.getenv("PORT", "5000"))
    app.run(debug=debug_mode, host="127.0.0.1", port=port)
