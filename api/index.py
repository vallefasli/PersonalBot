import os
import requests
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np

app = Flask(__name__)
CORS(app)

# --- CONFIGURATION ---
HF_TOKEN = os.getenv("HF_TOKEN", "your_huggingface_token_here") 
MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
# Official 2026 Router URL
API_URL = f"https://router.huggingface.co/hf-inference/models/{MODEL_ID}"
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

# 1. Load data
base_path = os.path.dirname(__file__)
csv_path = os.path.join(base_path, "PersonalBot_FineGrained.csv")
df = pd.read_csv(csv_path)

def get_embedding(text):
    """Fetches raw vector embedding from Hugging Face"""
    # Using the 'feature-extraction' task stops the 'missing argument: sentences' error
    payload = {
        "inputs": text,
        "task": "feature-extraction",
        "options": {"wait_for_model": True}
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    
    if response.status_code != 200:
        raise Exception(f"HF API Error: {response.text}")
    
    res_json = response.json()
    
    # Feature extraction returns a 3D list: [[[val1, val2, ...]]]
    # We convert to numpy and flatten it to a 1D vector
    vector = np.array(res_json)
    return vector.flatten()

# 2. Pre-embed the CSV (Runs on startup)
# WARNING: If your CSV has >30 rows, this might hit Vercel's 10s timeout.
print("Bot initializing: Encoding knowledge base...")
all_embeddings = np.array([get_embedding(t) for t in df['text'].tolist()])

@app.route("/api/chat", methods=["POST"])
def chat():
    try:
        data = request.json
        user_input = data.get("message", "")
        
        # --- KEYWORD FILTERING LOGIC ---
        mask = np.ones(len(df), dtype=bool)
        lower_input = user_input.lower()
        
        if "kpop" in lower_input or "bias" in lower_input:
            mask = (df['subtopic'] == "kpop") | (df['subtopic'] == "music")
        elif "anime" in lower_input:
            mask = (df['subtopic'] == "anime")
        elif "nba" in lower_input or "basketball" in lower_input:
            mask = (df['subtopic'] == "nba")
        elif "movie" in lower_input or "film" in lower_input:
            mask = (df['subtopic'] == "movies")
        elif "run" in lower_input or "outdoors" in lower_input:
            mask = (df['subtopic'] == "outdoors")
        elif "gunpla" in lower_input or "model" in lower_input:
            mask = (df['subtopic'] == "gunpla")

        filtered_df = df[mask]
        filtered_embeddings = all_embeddings[mask]

        # --- SIMILARITY LOGIC (NUMPY) ---
        user_vector = get_embedding(user_input)
        
        # Cosine Similarity
        dot_product = np.dot(filtered_embeddings, user_vector)
        norms = np.linalg.norm(filtered_embeddings, axis=1) * np.linalg.norm(user_vector)
        
        # Prevent division by zero
        scores = np.divide(dot_product, norms, out=np.zeros_like(dot_product), where=norms!=0)
        
        top_idx = np.argmax(scores)
        response = filtered_df.iloc[top_idx]['response']
        
        return jsonify({"response": response})
    
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"response": "SYSTEM_ERROR: Archive link failed. Check API Token."}), 500

if __name__ == "__main__":
    app.run()
