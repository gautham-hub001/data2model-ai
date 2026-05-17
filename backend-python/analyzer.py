import pandas as pd

def analyze_dataset(df):
    analysis = {}

    # Column types
    analysis["columns"] = {col: str(dtype) for col, dtype in df.dtypes.items()}

    # Missing values
    analysis["missing_values"] = df.isnull().sum().to_dict()

    # Basic stats
    analysis["describe"] = df.describe(include='all').fillna("").to_dict()

    # Correlation (numeric only)
    try:
        corr = df.corr(numeric_only=True)
        analysis["correlation"] = corr.to_dict()
    except:
        analysis["correlation"] = {}

    analysis["meta"] = extract_meta_features(df)
    return analysis

def extract_meta_features(df):
    return {
        "num_rows": df.shape[0],
        "num_cols": df.shape[1],
        "num_numeric": len(df.select_dtypes(include=["int64", "float64"]).columns),
        "num_categorical": len(df.select_dtypes(include=["object"]).columns),
        "has_missing": df.isnull().sum().sum() > 0
    }

def clean_dataset(df):
    # Convert TotalCharges to numeric if present
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    return df