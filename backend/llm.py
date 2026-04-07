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

print(f"✅ OpenRouter client initialized (Key Length: {len(api_key)})")


def generate_explanation(analysis, recommendation):
    prompt = f"""
You are a data scientist.

Analyze this dataset summary:
{analysis}

ML Recommendation:
{recommendation}

Provide a structured response:

1. Dataset Overview
2. Key Issues (missing values, correlations)
3. Recommended ML Approach (with reasoning)
4. Suggested Improvements

Keep it concise and professional.
"""

    response = client.chat.completions.create(
        model="meta-llama/llama-3-8b-instruct",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
        temperature=0.7
    )

    return response.choices[0].message.content

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
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content

def generate_code(df_columns, recommendation):
    prompt = f"""
    Generate Python sklearn code for this:

    Columns: {df_columns}
    Task: {recommendation}

    Include:
    - preprocessing
    - model training
    - evaluation
    """

    response = client.chat.completions.create(
        model="meta-llama/llama-3-8b-instruct",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=400
    )

    return response.choices[0].message.content