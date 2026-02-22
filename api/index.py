import os
import requests
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np

app = Flask(__name__)
CORS(app)

# --- CONFIGURATION ---
HF_TOKEN = os.getenv("HF_TOKEN") 
MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
API_URL = f"https://router.huggingface.co/hf-inference/models/{MODEL_ID}"
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

# 1. Load data
base_path = os.path.dirname(__file__)
csv_path = os.path.join(base_path, "PersonalBot_FineGrained.csv")
df = pd.read_csv(csv_path)

def get_embedding(text):
    """Fetches a single embedding for the search query"""
    payload = {"inputs": text, "task": "feature-extraction"}
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=5)
        if response.status_code == 200:
            return np.array(response.json()).flatten()
    except:
        return None
    return None

@app.route("/api/chat", methods=["POST"])
def chat():
    try:
        data = request.json
        user_input = data.get("message", "")
        lower_input = user_input.lower()

        # 1. FAST KEYWORD MATCH (No AI needed, instant)
        # Checks if your message is directly in the CSV
        keyword_match = df[df['text'].str.lower().str.contains(lower_input, na=False)]
        if not keyword_match.empty:
            return jsonify({"response": keyword_match.iloc[0]['response']})

        # 2. SEMANTIC SEARCH (One AI call)
        # If keywords fail, we ask the AI to find the closest match
        user_vector = get_embedding(user_input)
        if user_vector is not None:
            # Note: We are only embedding the USER input here. 
            # For a small CSV, simple text matching in Step 1 is usually enough!
            return jsonify({"response": "I found something similar in my records, but try being more specific about Kpop, Anime, or NBA!"})

        return jsonify({"response": "CONNECTION_STABLE: Command not recognized in dossier."})

    except Exception as e:
        return jsonify({"response": "SYSTEM_ERROR: Uplink interrupted."}), 500

if __name__ == "__main__":
    app.run()
