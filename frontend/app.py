import streamlit as st
import requests

st.title("📊 Data2Model AI")

uploaded_file = st.file_uploader("Upload your CSV", type=["csv"])

if uploaded_file:
    with st.spinner("Analyzing..."):
        files = {
            "file": (
                uploaded_file.name,
                uploaded_file.getvalue(),
                "text/csv"
            )
        }
        response = requests.post("https://data2model-ai.onrender.com/analyze", files=files)

        st.write("Status Code:", response.status_code)
        st.write("Raw Response:", response.text)

        if response.status_code != 200:
            st.error("Backend failed")
        else:
            data = response.json()
        st.subheader("📊 Analysis")
        st.json(data["analysis"])

        st.subheader("🤖 ML Recommendation")
        st.write(data["recommendation"])

        st.subheader("🧠 Explanation")
        st.write(data["explanation"])

        if st.button("Generate ML Code"):
            st.subheader("💻 Generated ML Code")
            st.code(data["code"], language="python")