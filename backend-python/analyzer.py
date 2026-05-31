import pandas as pd


def analyze_dataset(df):
    analysis = {}

    analysis["columns"] = {col: str(dtype) for col, dtype in df.dtypes.items()}
    analysis["missing_values"] = df.isnull().sum().to_dict()
    analysis["describe"] = df.describe(include='all').fillna("").to_dict()

    try:
        corr = df.corr(numeric_only=True)
        analysis["correlation"] = corr.to_dict()
    except Exception:
        analysis["correlation"] = {}

    analysis["meta"] = extract_meta_features(df)
    return analysis


def extract_meta_features(df):
    return {
        "num_rows": df.shape[0],
        "num_cols": df.shape[1],
        "num_numeric": len(df.select_dtypes(include=["int64", "float64"]).columns),
        "num_categorical": len(df.select_dtypes(include=["object"]).columns),
        "has_missing": df.isnull().sum().sum() > 0,
    }


def clean_dataset(df):
    # Coerce any column that looks numeric but was read as object (e.g. contains spaces or empty strings)
    for col in df.select_dtypes(include=["object"]).columns:
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().sum() > df[col].notna().sum() * 0.5:
            df = df.copy()
            df[col] = converted
    return df
