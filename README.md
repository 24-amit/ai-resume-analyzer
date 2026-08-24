# 📄 AI Resume Analyzer (RAG + LLM)

An end-to-end Retrieval-Augmented Generation (RAG) application that analyzes PDF resumes against job descriptions to produce match scores, skill gap breakdowns, and actionable recommendations.

Built with **FastAPI**, **ChromaDB**, **Streamlit**, and **Google Gemini API**.

---

## 🛠️ Tech Stack

* **Frontend:** [Streamlit](https://streamlit.io/)
* **Backend API:** [FastAPI](https://fastapi.tiangolo.com/) + Uvicorn
* **Vector Database:** [ChromaDB](https://www.trychroma.com/)
* **LLM:** [Google Gemini API](https://ai.google.dev/) (`gemini-3.6-flash`)
* **PDF Processing:** `pypdf`
* **Deployment:** [Render](https://render.com/)

---

## 🏗️ Architecture & How It Works

1. **Document Parsing:** Extracts raw text from uploaded PDF resumes.
2. **Chunking & Indexing:** Chunks resume text into semantically coherent segments and indexes them into temporary vector collections using **ChromaDB**.
3. **Semantic Retrieval (RAG):** Queries ChromaDB using the job description to extract the top-matching sections of the resume.
4. **LLM Evaluation:** Constructs a contextual prompt using the retrieved extracts and passes it to **Gemini API** for evaluation.
5. **Interactive UI:** Streamlit receives and renders the evaluation report in real time.

---

## 🚀 Local Setup & Installation

### Prerequisite
* Python 3.10+ installed
* Google Gemini API Key

### Step 1: Clone Repository
```bash
git clone https://github.com/24-amit/ai-resume-analyzer.git
cd ai-resume-analyzer
