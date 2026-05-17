import os
import io
import uuid
import functools

from flask import Flask, request, jsonify
import pandas as pd
from flask_cors import CORS
from supabase import create_client, Client

from analyzer import analyze_dataset, clean_dataset
from recommender import recommend_ml_task
from llm import generate_explanation, generate_code

app = Flask(__name__)
CORS(app)

INTERNAL_TOKEN = os.environ.get("INTERNAL_TOKEN", "")
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "")
STORAGE_BUCKET = "datasets"

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


def clean_for_json(obj):
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_for_json(v) for v in obj]
    elif hasattr(obj, "item"):
        return obj.item()
    return obj


def require_internal_token(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        if INTERNAL_TOKEN and request.headers.get("X-Internal-Token") != INTERNAL_TOKEN:
            return jsonify({"error": "Unauthorized"}), 401
        return f(*args, **kwargs)
    return wrapper


def _fetch_dataset(dataset_id: str) -> bytes:
    path = f"{dataset_id}.csv"
    return supabase.storage.from_(STORAGE_BUCKET).download(path)


def _run_analysis(df: pd.DataFrame, smote: bool) -> dict:
    if smote:
        from imblearn.over_sampling import SMOTE
        recommendation = recommend_ml_task(df)
        target_col = recommendation.get("target_variable")
        if target_col and target_col in df.columns:
            X = df.drop(columns=[target_col])
            y = df[target_col]
            X_numeric = X.select_dtypes(include="number")
            sm = SMOTE(random_state=42)
            X_res, y_res = sm.fit_resample(X_numeric, y)
            df = pd.concat([X_res, y_res.rename(target_col)], axis=1)

    analysis = analyze_dataset(df)
    recommendation = recommend_ml_task(df)
    explanation = generate_explanation(analysis, recommendation)
    code = generate_code(list(df.columns), recommendation)

    imbalance_detected = False
    target_col = recommendation.get("target_variable")
    if target_col and target_col in df.columns:
        class_counts = df[target_col].value_counts()
        if len(class_counts) >= 2:
            ratio = class_counts.min() / class_counts.max()
            imbalance_detected = ratio < 0.2

    return {
        "analysis": analysis,
        "recommendation": recommendation,
        "explanation": explanation,
        "code": code,
        "imbalanceDetected": imbalance_detected,
        "smoteApplied": smote,
    }


# ── Legacy endpoint ───────────────────────────────────────────────────────────

@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        file = request.files["file"]
        df = clean_dataset(pd.read_csv(file))
        return jsonify(clean_for_json(_run_analysis(df, smote=False)))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Endpoints called by Spring Boot ──────────────────────────────────────────

@app.route("/store-dataset", methods=["POST"])
@require_internal_token
def store_dataset():
    try:
        file = request.files["file"]
        dataset_id = str(uuid.uuid4())
        csv_bytes = file.read()
        supabase.storage.from_(STORAGE_BUCKET).upload(
            path=f"{dataset_id}.csv",
            file=csv_bytes,
            file_options={"content-type": "text/csv"}
        )
        return jsonify({"dataset_id": dataset_id})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/analyze-by-id", methods=["POST"])
@require_internal_token
def analyze_by_id():
    try:
        dataset_id = request.json.get("dataset_id")
        csv_bytes = _fetch_dataset(dataset_id)
        df = clean_dataset(pd.read_csv(io.BytesIO(csv_bytes)))
        return jsonify(clean_for_json(_run_analysis(df, smote=False)))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/analyze-smote-by-id", methods=["POST"])
@require_internal_token
def analyze_smote_by_id():
    try:
        dataset_id = request.json.get("dataset_id")
        csv_bytes = _fetch_dataset(dataset_id)
        df = clean_dataset(pd.read_csv(io.BytesIO(csv_bytes)))
        return jsonify(clean_for_json(_run_analysis(df, smote=True)))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "UP"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port)
