import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

# Get OpenRouter API key
api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    raise ValueError("CRITICAL: OPENROUTER_API_KEY is missing in .env")

# Initialize client with OpenRouter base URL
client = OpenAI(
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1"
)

model_name = "meta-llama/llama-3-8b-instruct"

print(f"✅ OpenRouter client initialized (Key Length: {len(api_key)})")



def generate_explanation(analysis, recommendation):
    prompt = f"""
    You are a data scientist.

    Analyze this dataset summary:
    {analysis}

    ML Recommendation:
    {recommendation}

    Explain:
    1. What this dataset is about
    2. Key issues (missing values, correlations)
    3. Why this ML approach is suitable
    4. Suggested improvements

    Keep it simple and structured.
    """

    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(override=True)

api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    raise ValueError("CRITICAL: OPENROUTER_API_KEY is missing in .env")

client = OpenAI(
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1",
    default_headers={
        "HTTP-Referer": "http://localhost:8501",
        "X-Title": "Data2Model AI"
    }
)

print(f"✅ OpenRouter client initialized")


def generate_explanation(analysis, recommendation):
    prompt = f"""
You are a data scientist.

Dataset Summary:
{analysis}

ML Recommendation:
{recommendation}

Provide:
1. Dataset Overview
2. Key Issues
3. Why this model is suitable
4. Improvements
"""

    response = client.chat.completions.create(
        model="meta-llama/llama-3-8b-instruct",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300
    )

    return response.choices[0].message.content


def generate_code(analysis, recommendation):
    prompt = f"""
You are generating production-ready sklearn code.

STRICT RULES:
- Use existing dataframe df (DO NOT create sample data)
- DO NOT truncate code
- Return COMPLETE runnable code

Dataset info:
{analysis}

Task:
{recommendation}

Requirements:
- Drop irrelevant columns (like customerID)
- Identify numeric columns (int64 + float64)
- Identify categorical columns
- Use ColumnTransformer
    - Numeric: SimpleImputer + StandardScaler
    - Categorical: SimpleImputer + OneHotEncoder
- Split train/test
- Train model
- Predict
- Print accuracy + classification report

IMPORTANT:
- DO NOT use LabelEncoder
- Return FULL working code
"""

    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=900
    )

    return response.choices[0].message.content

