from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer, util
import torch
import pandas as pd
import os

app = Flask(__name__)
CORS(app)

# 1. Load your data (Make sure the CSV is in the same 'api' folder)
# We use a path check to ensure Vercel finds the file
base_path = os.path.dirname(__file__)
csv_path = os.path.join(base_path, "PersonalBot_FineGrained.csv")
df = pd.read_csv(csv_path)

# 2. Load the model
model = SentenceTransformer('all-MiniLM-L6-v2')

# 3. Pre-embed the dataset (Crucial for speed on Vercel)
df['embedding'] = list(model.encode(df['text'].tolist(), convert_to_tensor=True))

@app.route("/api/chat", methods=["POST"])
def chat():
    try:
        data = request.json
        user_input = data.get("message", "")
        
        # --- YOUR EXACT KEYWORD FILTERING LOGIC ---
        filtered_df = df
        lower_input = user_input.lower()
        if "kpop" in lower_input or "bias" in lower_input:
            filtered_df = df[(df['subtopic'] == "kpop") | (df['subtopic'] == "music")]
        elif "anime" in lower_input:
            filtered_df = df[df['subtopic'] == "anime"]
        elif "nba" in lower_input or "basketball" in lower_input:
            filtered_df = df[df['subtopic'] == "nba"]
        elif "movie" in lower_input or "film" in lower_input:
            filtered_df = df[df['subtopic'] == "movies"]
        elif "run" in lower_input or "outdoors" in lower_input:
            filtered_df = df[df['subtopic'] == "outdoors"]
        elif "gunpla" in lower_input or "model" in lower_input:
            filtered_df = df[df['subtopic'] == "gunpla"]

        # --- YOUR EXACT SIMILARITY LOGIC ---
        user_emb = model.encode(user_input, convert_to_tensor=True)
        embeddings = torch.stack(filtered_df['embedding'].tolist())
        scores = util.cos_sim(user_emb, embeddings)[0]
        top_idx = torch.argmax(scores).item()
        
        response = filtered_df.iloc[top_idx]['response']
        
        return jsonify({"response": response})
    
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"response": "SYSTEM_ERROR: Database link unstable."}), 500

# Vercel requirements
if __name__ == "__main__":
    app.run()
