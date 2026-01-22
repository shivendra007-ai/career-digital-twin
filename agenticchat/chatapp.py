import os
import json
import tempfile
import uuid
from typing import TypedDict, List
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader
from langgraph.graph import StateGraph, END
import streamlit as st
#streamlit run agenticchat/chatapp.py

# ==============================================
# CONFIG
# ==============================================
load_dotenv(override=True)
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = "llama3.2"

client = OpenAI(base_url=OLLAMA_HOST + "/v1", api_key="ollama")

st.set_page_config(page_title="Career Digital Twin Chatbot", page_icon="💼")
st.title("💼 Career Digital Twin Chatbot (LLaMA 3.2)")
st.caption(f"Using model: {OLLAMA_MODEL} on {OLLAMA_HOST}")

# ==============================================
# LANGGRAPH NODES AND STRUCTURES
# ==============================================

class ResumeState(TypedDict):
    pdf_path: str
    resume_text: str
    analysis_result: dict

class JobMatchResult(BaseModel):
    best_job_titles: List[str]
    top_skills_matched: List[str]
    suggested_skills: List[str]
    career_path_summary: str

def extract_data(state: ResumeState) -> ResumeState:
    st.info("Extracting text from resume...")
    pdf_path = state["pdf_path"]
    resume_text = ""
    try:
        pdf_reader = PdfReader(pdf_path)
        for page in pdf_reader.pages:
            text = page.extract_text()
            if text:
                resume_text += text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        resume_text = "Error: Could not extract text."
    return {"resume_text": resume_text}

def analyze_career(state: ResumeState) -> ResumeState:
    st.info("Analyzing your career profile with LLaMA 3.2...")
    resume_text = state["resume_text"]
    if "Error:" in resume_text:
        return {"analysis_result": {"error": "Text extraction failed."}}
    
    schema = JobMatchResult.schema_json(indent=2)
    system_prompt = (
        f"You are an expert AI career advisor. "
        f"Analyze the given resume and respond ONLY in JSON, no explanations. "
        f"Schema:\n{schema}"
    )
    user_prompt = f"Analyze the following resume:\n\n{resume_text}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=OLLAMA_MODEL,
                messages=messages,
                response_format={"type": "json_object"}
            )
            raw = response.choices[0].message.content.strip().strip("`")
            if raw.startswith("json"):
                raw = raw[4:].strip()
            result = JobMatchResult.parse_raw(raw)
            return {"analysis_result": result.dict()}
        except Exception as e:
            if attempt == 2:
                return {"analysis_result": {"error": str(e)}} 

# Build the LangGraph workflow
workflow = StateGraph(ResumeState)
workflow.add_node("extract_data", extract_data)
workflow.add_node("analyze_career", analyze_career)
workflow.set_entry_point("extract_data")
workflow.add_edge("extract_data", "analyze_career")
workflow.add_edge("analyze_career", END)
app = workflow.compile()

# ==============================================
# STREAMLIT CHATBOT LOGIC
# ==============================================
if "resume_text" not in st.session_state:
    st.session_state.resume_text = ""
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = {}
if "messages" not in st.session_state:
    st.session_state.messages = []

# Greeting
if not st.session_state.messages:
    st.session_state.messages.append({
        "role": "assistant",
        "content": "👋 Hi! I'm your AI career advisor. Please upload your resume (PDF) to begin your personalized career analysis."
    })

uploaded_file = st.file_uploader("Upload your Resume (PDF)", type=["pdf"])

if uploaded_file:
    unique_filename = f"{uuid.uuid4()}_{uploaded_file.name}"
    temp_path = os.path.join(tempfile.gettempdir(), unique_filename)
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    if st.button("Analyze My Resume"):
        with st.spinner("Running analysis..."):
            initial_state = ResumeState(pdf_path=temp_path, resume_text="", analysis_result={})
            result_state = app.invoke(initial_state)
            st.session_state.resume_text = result_state["resume_text"]
            st.session_state.analysis_result = result_state["analysis_result"]
        
        st.success("✅ Resume analysis complete! You can now chat about your career options below.")

        os.remove(temp_path)

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input box
if prompt := st.chat_input("Ask me about your career, skills, or job opportunities..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if not st.session_state.resume_text:
            reply = "Please upload your resume first so I can understand your skills and background."
        else:
            # Build contextual prompt
            context = json.dumps(st.session_state.analysis_result, indent=2)
            user_message = (
                f"You are a friendly AI career coach. "
                f"Use the following analysis to answer naturally:\n{context}\n\nUser question: {prompt}"
            )
            response = client.chat.completions.create(
                model=OLLAMA_MODEL,
                messages=[
                    {"role": "system", "content": "You are a conversational AI career coach."},
                    {"role": "user", "content": user_message}
                ]
            )
            reply = response.choices[0].message.content

        st.markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})
