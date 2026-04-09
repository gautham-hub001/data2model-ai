# data2model-ai

This application lets User upload CSV and Backend analyzes the data and the LLM explains and suggests ML model and the UI shows structured output.

User uploads CSV in Streamlit
Streamlit sends file → Flask API
Flask:
Uses Pandas → analyze data
Uses logic → recommend ML task
Uses OpenAI → generate explanation
Response sent back → UI displays

Running the application:
Frontend (Streamlit):

1. cd frontend
2. Activate the venv: source venv/bin/activate
3. If needed, install the packages: pip install streamlit requests pandas plotly
4. streamlit run app.py
5. App should be running at http://localhost:8501/

Backend (Flask):

1. cd backend
2. python3 -m venv venv
3. source venv/bin/activate
4. pip install flask pandas numpy scikit-learn openai flask-cors gunicorn
5. Create OPENAI - API Key and store it in env variables:
   export OPENAI_API_KEY= "your_key"
6. Run app.py
7. It should be running at http://127.0.0.1:5000
