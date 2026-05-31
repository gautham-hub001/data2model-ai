import io
import logging
import os
import re
import threading
import uuid
from datetime import datetime, timedelta, timezone

import requests as http
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd

from analyzer import analyze_dataset, clean_dataset
from recommender import recommend_ml_task

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = Flask(__name__)

ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "http://localhost:3000")
CORS(app, origins=ALLOWED_ORIGINS.split(","))

INTERNAL_TOKEN = os.environ.get("INTERNAL_TOKEN", "")

SUPABASE_URL = re.sub(r'\s+', '', os.environ.get("SUPABASE_URL", ""))
SUPABASE_KEY = re.sub(r'\s+', '', os.environ.get("SUPABASE_SERVICE_KEY", ""))
STORAGE_BUCKET = "datasets"

MAX_FILE_BYTES = 10 * 1024 * 1024  # 10 MB
DATASET_TTL_HOURS = 24
_UUID_RE = re.compile(
    r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', re.IGNORECASE
)

if not SUPABASE_URL.startswith("https://"):
    raise RuntimeError(f"SUPABASE_URL looks wrong: {repr(SUPABASE_URL[:30])}")
if not SUPABASE_KEY.startswith("eyJ"):
    raise RuntimeError("SUPABASE_SERVICE_KEY looks wrong (should start with eyJ)")

_STORAGE_HEADERS = {
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "apikey": SUPABASE_KEY,
}


def _storage_url(path: str) -> str:
    return f"{SUPABASE_URL}/storage/v1/object/{STORAGE_BUCKET}/{path}"


def _upload_dataset(csv_bytes: bytes, dataset_id: str) -> None:
    url = _storage_url(f"{dataset_id}.csv")
    resp = http.post(url, data=csv_bytes, headers={**_STORAGE_HEADERS, "Content-Type": "text/csv"})
    resp.raise_for_status()


def _fetch_dataset(dataset_id: str) -> bytes:
    url = _storage_url(f"{dataset_id}.csv")
    resp = http.get(url, headers=_STORAGE_HEADERS)
    resp.raise_for_status()
    return resp.content


def _delete_dataset(dataset_id: str) -> None:
    url = _storage_url(f"{dataset_id}.csv")
    resp = http.delete(url, headers=_STORAGE_HEADERS)
    if resp.status_code not in (200, 404):
        log.warning("Failed to delete dataset %s: %s", dataset_id, resp.status_code)


def clean_for_json(obj):
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_for_json(v) for v in obj]
    elif hasattr(obj, "item"):
        return obj.item()
    return obj


def _validate_internal_token():
    if INTERNAL_TOKEN and request.headers.get("X-Internal-Token") != INTERNAL_TOKEN:
        return jsonify({"error": "Unauthorized"}), 401
    return None


def _validate_dataset_id(dataset_id: str):
    if not dataset_id or not _UUID_RE.match(dataset_id):
        return jsonify({"error": "Invalid dataset_id format."}), 400
    return None


def _run_analysis(df: pd.DataFrame, smote: bool) -> dict:
    if smote:
        from imblearn.over_sampling import SMOTE
        recommendation = recommend_ml_task(df)
        target_col = recommendation.get("target")
        if target_col and target_col in df.columns:
            X = df.drop(columns=[target_col])
            y = df[target_col]
            X_numeric = X.select_dtypes(include="number")
            sm = SMOTE(random_state=42)
            X_res, y_res = sm.fit_resample(X_numeric, y)
            df = pd.concat([X_res, y_res.rename(target_col)], axis=1)

    analysis = analyze_dataset(df)
    recommendation = recommend_ml_task(df)

    imbalance_detected = False
    target_col = recommendation.get("target")
    if target_col and target_col in df.columns:
        class_counts = df[target_col].value_counts()
        if len(class_counts) >= 2:
            ratio = class_counts.min() / class_counts.max()
            imbalance_detected = ratio < 0.2

    return {
        "analysis": analysis,
        "recommendation": recommendation,
        "imbalanceDetected": imbalance_detected,
        "smoteApplied": smote,
    }


def _cleanup_old_datasets():
    """Delete Supabase Storage objects whose sessions are older than DATASET_TTL_HOURS."""
    try:
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=DATASET_TTL_HOURS)).isoformat()
        headers = {
            "Authorization": f"Bearer {SUPABASE_KEY}",
            "apikey": SUPABASE_KEY,
        }
        resp = http.get(
            f"{SUPABASE_URL}/rest/v1/sessions",
            params={"select": "dataset_id", "created_at": f"lt.{cutoff}"},
            headers=headers,
        )
        if resp.status_code != 200:
            log.warning("Cleanup: failed to query sessions (%s)", resp.status_code)
            return

        rows = resp.json()
        for row in rows:
            did = row.get("dataset_id")
            if did and _UUID_RE.match(did):
                _delete_dataset(did)
                log.info("Cleanup: deleted dataset %s", did)

        log.info("Cleanup: processed %d expired datasets", len(rows))
    except Exception:
        log.exception("Dataset cleanup failed")


def _schedule_cleanup():
    _cleanup_old_datasets()
    timer = threading.Timer(DATASET_TTL_HOURS * 3600, _schedule_cleanup)
    timer.daemon = True
    timer.start()


# Start background cleanup scheduler
threading.Timer(60, _schedule_cleanup).daemon = True
threading.Timer(60, _schedule_cleanup).start()


# ── Legacy endpoint (direct file upload, no internal token required) ──────────

@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        file = request.files.get("file")
        if file is None:
            return jsonify({"error": "No file uploaded."}), 400

        if file.content_length and file.content_length > MAX_FILE_BYTES:
            return jsonify({"error": "File too large. Maximum size is 10 MB."}), 413

        filename = file.filename or ""
        if not filename.lower().endswith(".csv"):
            return jsonify({"error": "Only CSV files are accepted."}), 400

        raw = file.read(MAX_FILE_BYTES + 1)
        if len(raw) > MAX_FILE_BYTES:
            return jsonify({"error": "File too large. Maximum size is 10 MB."}), 413

        df = clean_dataset(pd.read_csv(io.BytesIO(raw)))
        return jsonify(clean_for_json(_run_analysis(df, smote=False)))
    except pd.errors.ParserError:
        return jsonify({"error": "Could not parse file as CSV."}), 400
    except Exception:
        log.exception("Error in /analyze")
        return jsonify({"error": "Analysis failed. Please check your file and try again."}), 500


# ── Endpoints called by Spring Boot ──────────────────────────────────────────

@app.route("/store-dataset", methods=["POST"])
def store_dataset():
    err = _validate_internal_token()
    if err:
        return err
    try:
        file = request.files.get("file")
        if file is None:
            return jsonify({"error": "No file provided."}), 400

        raw = file.read(MAX_FILE_BYTES + 1)
        if len(raw) > MAX_FILE_BYTES:
            return jsonify({"error": "File too large. Maximum size is 10 MB."}), 413

        filename = file.filename or ""
        if not filename.lower().endswith(".csv"):
            return jsonify({"error": "Only CSV files are accepted."}), 400

        dataset_id = str(uuid.uuid4())
        _upload_dataset(raw, dataset_id)
        return jsonify({"dataset_id": dataset_id})
    except Exception:
        log.exception("Error in /store-dataset")
        return jsonify({"error": "Failed to store dataset."}), 500


@app.route("/analyze-by-id", methods=["POST"])
def analyze_by_id():
    err = _validate_internal_token()
    if err:
        return err
    try:
        dataset_id = (request.json or {}).get("dataset_id", "").strip()
        err = _validate_dataset_id(dataset_id)
        if err:
            return err

        csv_bytes = _fetch_dataset(dataset_id)
        df = clean_dataset(pd.read_csv(io.BytesIO(csv_bytes)))
        return jsonify(clean_for_json(_run_analysis(df, smote=False)))
    except pd.errors.ParserError:
        return jsonify({"error": "Stored file is not a valid CSV."}), 400
    except Exception:
        log.exception("Error in /analyze-by-id")
        return jsonify({"error": "Analysis failed."}), 500


@app.route("/analyze-smote-by-id", methods=["POST"])
def analyze_smote_by_id():
    err = _validate_internal_token()
    if err:
        return err
    try:
        dataset_id = (request.json or {}).get("dataset_id", "").strip()
        err = _validate_dataset_id(dataset_id)
        if err:
            return err

        csv_bytes = _fetch_dataset(dataset_id)
        df = clean_dataset(pd.read_csv(io.BytesIO(csv_bytes)))
        return jsonify(clean_for_json(_run_analysis(df, smote=True)))
    except pd.errors.ParserError:
        return jsonify({"error": "Stored file is not a valid CSV."}), 400
    except Exception:
        log.exception("Error in /analyze-smote-by-id")
        return jsonify({"error": "Analysis with SMOTE failed."}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "UP"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port)
