import streamlit as st
import requests

st.set_page_config(page_title="AI Resume Analyzer", layout="centered")

st.title("📄 AI Resume Analyzer (RAG + LLM)")
st.write("Upload a PDF resume and target job description to get match scores and recommendations.")

uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
job_description = st.text_area("Paste Job Description", height=200, placeholder="Paste requirements, skills, and responsibilities...")

if st.button("Analyze Resume", type="primary"):
    if not uploaded_file or not job_description.strip():
        st.error("Please provide both a PDF resume and a Job Description.")
    else:
        with st.spinner("Processing embeddings and analyzing alignment..."):
            try:
                files = {"resume": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                data = {"job_description": job_description}
                
                response = requests.post("http://127.0.0.1:8000/analyze", files=files, data=data)
                
                if response.status_code == 200:
                    result = response.json()
                    st.subheader("📊 Analysis & Improvement Report")
                    st.markdown(result["analysis"])
                else:
                    # Print the raw response text if JSON parsing fails
                    try:
                        error_msg = response.json().get("detail", response.text)
                    except Exception:
                        error_msg = response.text
                    st.error(f"Backend Error ({response.status_code}): {error_msg}")
            except requests.exceptions.ConnectionError:
                st.error("Could not connect to FastAPI backend. Ensure it is running on port 8000.")