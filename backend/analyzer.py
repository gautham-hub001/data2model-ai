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

    return analysis