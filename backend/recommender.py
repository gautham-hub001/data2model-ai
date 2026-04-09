import pandas as pd

# -------------------------
# PROBLEM TYPE
# -------------------------
def detect_problem_type(df, target):
    unique = df[target].nunique()

    if df[target].dtype == "object" or unique <= 15:
        return "classification"
    else:
        return "regression"


# -------------------------
# META FEATURES
# -------------------------
def extract_meta_features(df, target):
    return {
        "n_samples": int(df.shape[0]),
        "n_features": int(df.shape[1] - 1),
        "n_numeric": int(len(df.select_dtypes(include=["int64", "float64"]).columns)),
        "n_categorical": int(len(df.select_dtypes(include=["object"]).columns)),
        "missing_ratio": float(df.isnull().sum().sum() / df.size),
    }


# -------------------------
# IMBALANCE
# -------------------------
def detect_imbalance(df, target):
    counts = df[target].value_counts(normalize=True)

    if len(counts) <= 1:
        return False

    return bool(counts.max() > 0.75)


# -------------------------
# CORRELATION STRENGTH
# -------------------------
def get_correlation_strength(df, target):
    corr = df.corr(numeric_only=True)

    if target not in corr:
        return 0

    target_corr = corr[target].drop(labels=[target], errors="ignore")

    if target_corr.empty:
        return 0

    return float(abs(target_corr).max())


# -------------------------
# DATA TYPE DETECTION
# -------------------------
def detect_data_type(meta):
    if meta["n_features"] > 100:
        return "high_dimensional"
    elif meta["n_categorical"] > 0:
        return "tabular"
    else:
        return "numeric"


# -------------------------
# TARGET DETECTION
# -------------------------
def detect_target(df):
    candidates = []

    for col in df.columns:
        unique = df[col].nunique()
        ratio = unique / len(df)

        # Skip ID-like columns
        if "id" in col.lower() or "invoice" in col.lower():
            continue

        # Good classification target:
        # - few unique values
        # - not too unique relative to dataset size
        if unique <= 10 and ratio < 0.05:
            candidates.append((col, unique))

    # Return best candidate
    if candidates:
        candidates.sort(key=lambda x: x[1])
        return candidates[0][0]

    # ❗ IMPORTANT: No target found
    return None

def detect_unsupervised(df):
    n_rows = len(df)
    n_cols = df.shape[1]

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    categorical_cols = df.select_dtypes(include=["object"]).columns

    avg_unique_ratio = sum(df[col].nunique() / n_rows for col in df.columns) / n_cols

    # High repetition → transactional pattern
    if avg_unique_ratio < 0.1:
        return {
            "model": "Apriori",
            "target": None,
            "reason": "High repetition dataset; suitable for association rule mining"
        }

    # Otherwise → clustering
    return {
        "model": "KMeans",
        "target": None,
        "reason": "No target variable; clustering recommended"
    }

# -------------------------
# MAIN RULE ENGINE
# -------------------------
def choose_model(df, target):
    meta = extract_meta_features(df, target)

    problem = detect_problem_type(df, target)
    imbalance = detect_imbalance(df, target)
    corr_strength = get_correlation_strength(df, target)
    data_type = detect_data_type(meta)

    n_samples = meta["n_samples"]
    n_features = meta["n_features"]

    print("DEBUG:")
    print("n_samples:", n_samples)
    print("n_features:", n_features)
    print("corr_strength:", corr_strength)
    print("imbalance:", imbalance)
    print("data_type:", data_type)

    # -------------------------
    # CLASSIFICATION
    # -------------------------
    if problem == "classification":

        # High dimensional (text-like)
        if data_type == "high_dimensional":
            return {
                "model": "Linear SVC",
                "reason": "High-dimensional sparse data"
            }

        # Imbalanced data
        if imbalance:
            return {
                "model": "Random Forest",
                "reason": "Handles class imbalance"
            }

        # Strong non-linear signal
        if corr_strength > 0.5:
            return {
                "model": "Random Forest",
                "reason": "Strong feature-target relationships detected"
            }

        # Very small dataset
        if n_samples < 300:
            return {
                "model": "Logistic Regression",
                "reason": "Small dataset"
            }

        # Tabular data
        if n_features <= 50:
            return {
                "model": "Random Forest",
                "reason": "Best for structured tabular data"
            }

        # Large dataset
        if n_samples > 10000:
            return {
                "model": "Gradient Boosting",
                "reason": "Large dataset, better accuracy"
            }

        return {
            "model": "Random Forest",
            "reason": "General-purpose classifier"
        }

    # -------------------------
    # REGRESSION
    # -------------------------
    else:

        if corr_strength > 0.7:
            return {
                "model": "Linear Regression",
                "reason": "Strong linear relationship"
            }

        if n_features <= 50:
            return {
                "model": "Random Forest Regressor",
                "reason": "Non-linear tabular data"
            }

        if n_samples > 10000:
            return {
                "model": "Gradient Boosting Regressor",
                "reason": "Large dataset"
            }

        return {
            "model": "Linear Regression",
            "reason": "Fallback regression model"
        }


# -------------------------
# MAIN ENTRY
# -------------------------
def recommend_ml_task(df):
    target = detect_target(df)

    if target is None:
        return detect_unsupervised(df)

    model_info = choose_model(df, target)

    return {
        "model": model_info["model"],
        "target": target,
        "reason": model_info["reason"]
    }