def detect_target(df):
    for col in df.columns:
        if col.lower() in ["churn", "target", "label"]:
            return col
    return df.columns[-1]


def detect_imbalance(df, target_col):
    counts = df[target_col].value_counts(normalize=True)

    if len(counts) <= 1:
        return False

    return bool(counts.max() > 0.8)  # ✅ FIX


def extract_dataset_features(df):
    return {
        "num_rows": int(df.shape[0]),
        "num_cols": int(df.shape[1]),
        "num_numeric": int(len(df.select_dtypes(include=["int64", "float64"]).columns)),
        "num_categorical": int(len(df.select_dtypes(include=["object"]).columns)),
        "has_missing": bool(df.isnull().sum().sum() > 0)  # ✅ FIX
    }


def choose_model_advanced(df, target_col, meta):
    n_samples = meta["num_rows"]
    n_features = meta["num_cols"]

    imbalance = detect_imbalance(df, target_col)
    target_unique = df[target_col].nunique()

    is_classification = (
        df[target_col].dtype == "object" or target_unique <= 10
    )

    if is_classification:
        if imbalance:
            model = "Random Forest"
            reason = "Handles class imbalance well"

        elif n_features > 50:
            model = "SGDClassifier"
            reason = "High-dimensional dataset"

        elif n_samples < 1000:
            model = "Logistic Regression"
            reason = "Small dataset"

        else:
            model = "Random Forest"
            reason = "Balanced medium dataset"

    else:
        if n_features > 50:
            model = "SGDRegressor"
            reason = "High-dimensional data"

        elif n_samples < 1000:
            model = "Linear Regression"
            reason = "Small dataset"

        else:
            model = "Random Forest Regressor"
            reason = "General purpose model"

    return {
        "model": model,
        "target": target_col,
        "reason": reason
    }


def recommend_ml_task(df):
    target_col = detect_target(df)
    meta = extract_dataset_features(df)

    return choose_model_advanced(df, target_col, meta)