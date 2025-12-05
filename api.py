import time
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS
from textblob import TextBlob
import textstat

app = Flask(__name__)

# FIX CORS PROPERLY - Allow all origins
CORS(app, resources={
    r"/api/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})
@app.after_request
def apply_cors(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response

# Load models
try:
    model = joblib.load('logreg_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    print("--- Models and Vectorizer loaded successfully! ---")
except Exception as e:
    print(f"--- ERROR LOADING MODELS: {e} ---")
    model = None
    vectorizer = None

@app.route('/api/')
def home():
    return 'Fake News Detector API is running!'

@app.route('/api/analyze', methods=['POST', 'OPTIONS'])
def predict_news():
    # Handle preflight
    if request.method == 'OPTIONS':
        return '', 204
        
    if not model or not vectorizer:
        return jsonify({"error": "Models not loaded"}), 500

    data = request.get_json(force=True)
    text = data.get('text', '')

    if not text.strip():
        return jsonify({"error": "No text provided"}), 400

    try:
        # Prediction
        x_new = vectorizer.transform([text])
        probs = model.predict_proba(x_new)[0]
        prediction = probs.argmax()
        label = "FAKE" if prediction == 1 else "REAL"
        confidence = float(max(probs))

        # Analysis
        word_count = len(text.split())
        blob = TextBlob(text)
        sentiment_polarity = blob.sentiment.polarity
        
        if sentiment_polarity > 0.02:
            sentiment = "Positive"
        elif sentiment_polarity < -0.02:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"

        readability_score = textstat.flesch_reading_ease(text)
        keywords = []

        result = {
            "prediction": label,
            "confidence": confidence,
            "probability": {
                "fake": float(probs[1]),
                "real": float(probs[0])
            },
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "analysis": {
                "sentiment": sentiment,
                "keywords": keywords,
                "readability_score": readability_score,
                "word_count": word_count
            }
        }
        
        return jsonify(result)

    except Exception as e:
        print(f"--- PREDICTION ERROR: {e} ---")
        return jsonify({"error": f"Error: {e}"}), 500

@app.route('/api/health')
def health_check():
    return {"status": "ok"}, 200

if __name__ == '__main__':
    app.run(debug=True, port=5000)



