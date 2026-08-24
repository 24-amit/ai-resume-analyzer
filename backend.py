import os
import io
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pypdf import PdfReader
import chromadb
from google import genai
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="AI Resume Analyzer API")

# Initialize client using GEMINI_API_KEY from .env
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
chroma_client = chromadb.Client()

def extract_text_from_pdf(file_bytes: bytes) -> str:
    pdf = PdfReader(io.BytesIO(file_bytes))
    text = ""
    for page in pdf.pages:
        text += page.extract_text() or ""
    return text.strip()

@app.post("/analyze")
async def analyze_resume(
    resume: UploadFile = File(...),
    job_description: str = Form(...)
):
    if not resume.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    # 1. Extract Text
    content = await resume.read()
    resume_text = extract_text_from_pdf(content)
    if not resume_text:
        raise HTTPException(status_code=400, detail="Could not extract text from PDF.")

    # 2. Vector Search via ChromaDB
    collection_name = f"resume_eval_{os.urandom(4).hex()}"
    collection = chroma_client.create_collection(name=collection_name)
    
    chunks = [c.strip() for c in resume_text.split("\n\n") if len(c.strip()) > 30]
    if not chunks:
        chunks = [resume_text]

    collection.add(
        documents=chunks,
        ids=[f"chunk_{i}" for i in range(len(chunks))]
    )

    query_results = collection.query(query_texts=[job_description], n_results=min(3, len(chunks)))
    relevant_context = "\n".join(query_results["documents"][0]) if query_results["documents"] else resume_text[:1500]

    # Cleanup ChromaDB temporary collection
    chroma_client.delete_collection(name=collection_name)

    # 3. Call Gemini Model
    prompt = f"""
You are an expert ATS system and technical recruiter. Compare the candidate's resume context against the Job Description.

Job Description:
{job_description}

Relevant Resume Extracts:
{relevant_context}

Provide a structured analysis:
1. Match Score (0-100%)
2. Key Strengths
3. Missing Skills / Gaps
4. Actionable Recommendations
"""

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt
    )

    return {
        "status": "success",
        "analysis": response.text
    }