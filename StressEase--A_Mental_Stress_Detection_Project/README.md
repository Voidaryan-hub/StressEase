# AI-Powered Stress and Mood Detection System

StressEase AI is a Flask-based mental health monitoring prototype for students. It includes:

- AI stress detection using the trained `model_rf.pkl` model
- DASS-21 and PSS-10 questionnaire scoring
- Mood detection from journal text
- OpenCV-based webcam facial expression detection
- FER2013 CNN model hook through `FER_CNN_MODEL_PATH`
- Monitoring dashboard with latest stress and mood signals
- Mood calendar visualization
- AI wellness chat with local fallback replies
- Personalized recommendations based on risk level
- SQLite login/signup authentication
- SQLite storage in `StressEase.db`

## Run Locally

```bash
cd ~/Downloads/StressEase--/StressEase--A_Mental_Stress_Detection_Project
source .venv314/bin/activate
PORT=5001 python app.py
```

Open:

```text
http://127.0.0.1:5001
```

Use `PORT=5001` because macOS AirPlay Receiver often uses port `5000`.

## Note

This is a student project prototype and not a medical diagnosis tool.
