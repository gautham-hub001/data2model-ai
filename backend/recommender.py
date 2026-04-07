def recommend_ml_task(df):
    # PRIORITIZE churn-like columns
    for col in df.columns:
        if col.lower() in ["churn", "target", "label"]:
            return f"Classification (target: {col})"

    for col in df.columns:
        unique_vals = df[col].nunique()

        if df[col].dtype == 'object' and unique_vals <= 10:
            return f"Classification (target: {col})"

    return "Regression or Clustering"