import time
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS
from textblob import TextBlob  # <-- 1. Import TextBlob
import textstat              # <-- 2. Import textstat

app = Flask(__name__)
app.config['CORS_HEADERS'] = 'Content-Type'
CORS(app, resources={r"/api/*": {"origins": "*"}})
# --- Load your trained models ---
try:
    model = joblib.load('logreg_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    print("--- Models and Vectorizer loaded successfully! ---")
except Exception as e:
    print(f"--- ERROR LOADING MODELS: {e} ---")
    model = None
    vectorizer = None

# --- Home route ---
@app.route('/api/')
def home():
    return 'Fake News Detector API is running!'

# --- Prediction route ---
@app.route('/api/analyze', methods=['POST'])
def predict_news():
    if not model or not vectorizer:
        return jsonify({"error": "Models not loaded, check server logs"}), 500

    data = request.get_json(force=True)
    text = data.get('text', '')  # Get the text from the JS request

    if not text.strip():
        return jsonify({"error": "No text provided"}), 400

    try:
        # --- 1. Your AI Model Prediction ---
        x_new = vectorizer.transform([text])
        probs = model.predict_proba(x_new)[0]
        prediction = probs.argmax()

        # Map 0/1 to labels (adjust if your model is different)
        # (0 = real, 1 = fake)
        label = "FAKE" if prediction == 1 else "REAL"
        confidence = float(max(probs))

        # --- 2. NEW: Perform Analysis ---
        
        # Word Count
        word_count = len(text.split())

        # Sentiment
        blob = TextBlob(text)
        sentiment_polarity = blob.sentiment.polarity
        if sentiment_polarity > 0.02:
            sentiment = "Positive"
        elif sentiment_polarity < -0.02:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"

        # Readability Score
        readability_score = textstat.flesch_reading_ease(text)
        
        # Keywords (This is a placeholder. Real keyword extraction is complex)
        keywords = [] 

        # --- 3. Format the Result ---
        result = {
            "prediction": label,
            "confidence": confidence,  # <-- FIX 1: Send the raw number, not a string
            "probability": {
                "fake": float(probs[1]),
                "real": float(probs[0])
            },
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            
            # <-- FIX 2: Added the analysis object your frontend wants
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
        return jsonify({"error": f"An error occurred during prediction: {e}"}), 500

# --- Health check route ---
@app.route('/api/health')
def health_check():
    return {"status": "ok"}, 200

# --- Run the app ---
if __name__ == '__main__':

    app.run(debug=True, port=5000)

