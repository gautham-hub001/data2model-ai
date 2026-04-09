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
        #response = requests.post("https://localhost:5001/analyze", files=files) # For local testing


        st.write("Status Code:", response.status_code)

        if response.status_code != 200:
            st.error("Backend failed")
            with st.expander("🔍 View Raw Error"):
                st.text(response.text)
        else:
            data = response.json()
            st.success("✅ Analysis Complete")

            # Optional debug section
            with st.expander("🔍 View Raw Response (Debug)"):
                st.json(data)
        
        st.subheader("📊 Analysis")
        st.json(data["analysis"])

        st.subheader("🤖 ML Recommendation")
        st.write(data["recommendation"])

        st.subheader("🧠 Explanation")
        st.write(data["explanation"])

        if st.button("Generate ML Code"):
            st.subheader("💻 Generated ML Code")
            st.code(data["code"], language="python")