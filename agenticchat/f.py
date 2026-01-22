import os
import json
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader
from pydantic import BaseModel, Field
from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, END
import tempfile
import uuid
import time
#python -m streamlit run agenticchat\f.py
# ---------------------------------------------
# 1️⃣ CONFIGURATION
# ---------------------------------------------
load_dotenv(override=True)

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = "llama3.2"

client = OpenAI(
    base_url=f"{OLLAMA_HOST}/v1",
    api_key="ollama"
)
st.caption(f"💻 Using Local LLM: **{OLLAMA_MODEL}** hosted at `{OLLAMA_HOST}`")


# ---------------------------------------------
# 2️⃣ STATE & OUTPUT SCHEMA
# ---------------------------------------------
class ResumeState(TypedDict):
    pdf_path: str
    resume_text: str
    analysis_result: dict


class JobMatchResult(BaseModel):
    best_job_titles: List[str] = Field(description="Top 3 job titles most aligned with the resume")
    top_skills_matched: List[str] = Field(description="Top 5 skills found in the resume")
    suggested_skills: List[str] = Field(description="Top 5 in-demand skills to learn next")
    career_path_summary: str = Field(description="Brief and motivating summary")
    confidence_score: Optional[float] = Field(default=0.8, description="Confidence (0–1) in the analysis accuracy")
    recommended_industries: Optional[List[str]] = Field(default=[], description="Industries most aligned with the user profile")


# ---------------------------------------------
# 3️⃣ GRAPH NODES (LangGraph Workflow)
# ---------------------------------------------
def extract_data(state: ResumeState) -> ResumeState:
    """Node 1: Extract resume text."""
    st.info("📄 Extracting text from your resume...")
    resume_text = ""
    pdf_path = state["pdf_path"]
    try:
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            content = page.extract_text()
            if content:
                resume_text += content + "\n"
    except Exception as e:
        st.error(f"❌ PDF reading error: {e}")
        resume_text = "Error: Unable to extract text from PDF."
    return {"resume_text": resume_text}


def analyze_career(state: ResumeState) -> ResumeState:
    """Node 2: Run the LLM to analyze the extracted resume text."""
    st.info("🤖 Analyzing career fit using LLaMA 3.2...")
    resume_text = state["resume_text"]

    if "Error" in resume_text:
        return {"analysis_result": {"error": "Text extraction failed."}}

    json_schema = JobMatchResult.schema_json(indent=2)

    system_prompt = (
        "You are an AI Career Consultant creating a 'Career Digital Twin'. "
        "Analyze the resume below and generate an inspiring, structured output. "
        "Return ONLY valid JSON according to the schema — no extra text, markdown, or commentary.\n\n"
        f"JSON Schema:\n{json_schema}"
    )

    user_prompt = (
        f"Analyze the following resume and identify:\n"
        f"- Top 3 job roles most aligned with the candidate\n"
        f"- Top 5 current skills found\n"
        f"- Top 5 future skills to learn\n"
        f"- Recommended industries\n"
        f"- Confidence score (0–1)\n"
        f"- Short motivational career summary\n\n"
        f"--- RESUME ---\n{resume_text}"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=OLLAMA_MODEL,
                messages=messages,
                response_format={"type": "json_object"},
            )

            result = response.choices[0].message.content.strip()
            if result.startswith("```"):
                result = result.split("```")[-2].strip()

            parsed = JobMatchResult.parse_raw(result)
            st.success(f"✅ Analysis completed on attempt {attempt + 1}")
            return {"analysis_result": parsed.dict()}
        except Exception as e:
            st.warning(f"⚠️ Retry {attempt + 1}/3 failed — {e}")
            time.sleep(1)
            if attempt == 2:
                st.error("❌ Could not parse LLM output after 3 attempts.")
                return {"analysis_result": {"error": str(e)}}


# ---------------------------------------------
# 4️⃣ BUILD LANGGRAPH PIPELINE
# ---------------------------------------------
workflow = StateGraph(ResumeState)
workflow.add_node("extract_data", extract_data)
workflow.add_node("analyze_career", analyze_career)

workflow.set_entry_point("extract_data")
workflow.add_edge("extract_data", "analyze_career")
workflow.add_edge("analyze_career", END)
app = workflow.compile()


# ---------------------------------------------
# 5️⃣ STREAMLIT UI
# ---------------------------------------------
st.title("🧠 Career Digital Twin — Agentic AI (LangGraph + LLaMA 3.2)")
st.markdown("Upload your resume to generate a **personalized career analysis** powered by your **local LLaMA 3.2 model**.")

uploaded_file = st.file_uploader("📎 Upload Resume (PDF)", type=["pdf"])

if uploaded_file:
    temp_filename = str(uuid.uuid4()) + "_" + uploaded_file.name
    temp_path = os.path.join(tempfile.gettempdir(), temp_filename)
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    if st.button("🚀 Run Career Analysis"):
        with st.spinner("Processing through LangGraph pipeline..."):
            init_state = ResumeState(pdf_path=temp_path, resume_text="", analysis_result={})
            try:
                result_state = app.invoke(init_state)
                output = result_state["analysis_result"]

                if "error" in output:
                    st.error(f"⚠️ Error: {output['error']}")
                else:
                    st.markdown("---")
                    st.header("🏆 Your Personalized Career Report")

                    st.subheader("🎯 Top Job Roles")
                    st.markdown("\n".join([f"- **{job}**" for job in output.get("best_job_titles", [])]))

                    st.subheader("🏢 Recommended Industries")
                    if output.get("recommended_industries"):
                        st.markdown(", ".join(output["recommended_industries"]))
                    else:
                        st.markdown("_Not enough data to recommend industries._")

                    st.subheader("📈 Skills Overview")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Top Skills Found:**")
                        st.markdown("\n".join([f"✅ {s}" for s in output.get("top_skills_matched", [])]))
                    with col2:
                        st.markdown("**Future Skills to Learn:**")
                        st.markdown("\n".join([f"🚀 {s}" for s in output.get("suggested_skills", [])]))

                    st.subheader("💡 Summary")
                    st.info(output.get("career_path_summary", "No summary available."))

                    st.progress(float(output.get("confidence_score", 0.7)))
                    st.caption(f"Confidence Score: **{output.get('confidence_score', 0.7):.2f}**")

            except Exception as e:
                st.exception(e)
            finally:
                os.remove(temp_path)
