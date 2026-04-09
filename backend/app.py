from flask import Flask, request, jsonify
import pandas as pd
from analyzer import analyze_dataset
from recommender import recommend_ml_task
from llm import generate_explanation, generate_code
from flask_cors import CORS
from analyzer import clean_dataset

print("--- 🏁 APP.PY IS STARTING ---")  # Add this at line 1

from flask import Flask, request, jsonify
print("--- Flask Loaded ---")

import pandas as pd
print("--- Pandas Loaded ---")

from llm import generate_explanation  # Move this UP
print("--- LLM module Loaded ---")

# ... rest of your imports

app = Flask(__name__)
CORS(app)

def clean_for_json(obj):
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_for_json(v) for v in obj]
    elif hasattr(obj, "item"):  # handles numpy types
        return obj.item()
    else:
        return obj
    
@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        print("FILES RECEIVED:", request.files)
        file = request.files["file"]

        df = pd.read_csv(file)
        df = clean_dataset(df)

        analysis = analyze_dataset(df)
        recommendation = recommend_ml_task(df)
        explanation = generate_explanation(analysis, recommendation)

        code = generate_code(list(df.columns), recommendation)

        response_dict = {
            "analysis": analysis,
            "recommendation": recommendation,
            "explanation": explanation,
            "code": code
        }

        return jsonify(clean_for_json(response_dict))

    except Exception as e:
        print("ERROR:", str(e))  # 👈 shows in terminal
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5001)

