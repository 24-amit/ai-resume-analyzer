import os
import io
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pypdf import PdfReader
from google import genai
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="AI Resume Analyzer API")

# Initialize Gemini Client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def extract_text_from_pdf(file_bytes: bytes) -> str:
    pdf = PdfReader(io.BytesIO(file_bytes))
    extracted_text = ""
    # Dynamically extract text across all pages (handles 1, 2, or 5+ pages)
    for page in pdf.pages:
        text = page.extract_text()
        if text:
            extracted_text += text + "\n"
    return extracted_text.strip()

@app.get("/")
def health_check():
    return {"status": "Backend is running"}

@app.post("/analyze")
async def analyze_resume(
    resume: UploadFile = File(...),
    job_description: str = Form(...)
):
    if not resume.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    # 1. Extract Text from PDF (All Pages)
    content = await resume.read()
    resume_text = extract_text_from_pdf(content)
    if not resume_text:
        raise HTTPException(status_code=400, detail="Could not extract text from PDF.")

    # 2. Prompt Gemini Model
    prompt = f"""
You are an expert ATS system and technical recruiter. Compare the candidate's resume context against the Job Description regardless of the resume length or page count.

Job Description:
{job_description}

Complete Resume Text:
{resume_text}

Provide a structured analysis:
1. Match Score (0-100%)
2. Key Strengths
3. Missing Skills / Gaps
4. Actionable Recommendations
"""

    try:
        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt
        )

        return {
            "status": "success",
            "analysis": response.text
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))