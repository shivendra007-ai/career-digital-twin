import os
import json
import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader
from pydantic import BaseModel, Field
import tempfile
from typing import List

# --- CONFIGURATION & SETUP ---
load_dotenv(override=True)
ollama_host = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
ollama_model = "llama3.2"

# Ollama Client configured for the local server
client = OpenAI(
    base_url=ollama_host + "/v1",
    api_key="ollama"
)
print(f"Ollama Client configured for: {ollama_host} using model: {ollama_model}")

# --- 1. DATA STRUCTURES ---

class InitialSummary(BaseModel):
    summary: str = Field(description="A concise professional summary based on the resume.")
    suggested_roles: List[str] = Field(description="A list of 3 best-fit job titles for the candidate.")


# --- 2. CORE FUNCTIONS ---

def extract_text_from_pdf(file_path):
    """Extract all text from a PDF file."""
    try:
        reader = PdfReader(file_path)
        resume_text = "".join(page.extract_text() for page in reader.pages if page.extract_text())
        if not resume_text:
            raise ValueError("Could not extract text from PDF.")
        return resume_text
    except Exception as e:
        return f"ERROR: Failed to process PDF. {e}"


def generate_initial_summary(resume_text: str) -> str:
    """Generates the initial structured context summary for the chat agent."""
    initial_prompt = (
        "You are an expert HR Analyst. Analyze the following resume text and provide "
        "a concise professional summary and 3 best-fit job titles. "
        "Respond in raw JSON ONLY using this format:\n"
        "{"
        "\"summary\": \"...\", "
        "\"suggested_roles\": [\"...\", \"...\", \"...\"]"
        "}"
    )

    user_message = f"RESUME TEXT:\n{resume_text}"

    messages = [
        {"role": "system", "content": initial_prompt},
        {"role": "user", "content": user_message}
    ]

    try:
        response = client.chat.completions.create(
            model=ollama_model,
            messages=messages
        )

        json_output = response.choices[0].message.content.strip()
        if json_output.startswith("```json"):
            json_output = json_output[7:-3].strip()
        elif json_output.startswith("```"):
            json_output = json_output[3:-3].strip()

        summary_data = InitialSummary.parse_raw(json_output)

        context = (
            f"PROFESSIONAL SUMMARY: {summary_data.summary}\n"
            f"BEST FIT ROLES: {', '.join(summary_data.suggested_roles)}\n"
            f"RAW RESUME TEXT:\n{resume_text}"
        )
        return context

    except Exception as e:
        print(f"⚠️ LLM Processing Error: {e}")
        print(f"Raw output was: {json_output}")
        return f"ERROR: LLM failed to generate or parse initial summary. {e}"


def handle_file_upload(file, gr_state):
    """Handles the file upload and processes the resume."""
    if file is None:
        return gr_state, "Please upload a PDF file.", ""

    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, file.name if file.name else "uploaded_resume.pdf")

    file.seek(0)
    with open(temp_path, "wb") as f:
        f.write(file.read())

    resume_text = extract_text_from_pdf(temp_path)
    os.remove(temp_path)

    if "ERROR" in resume_text:
        return gr_state, resume_text, None

    llm_context = generate_initial_summary(resume_text)

    if "ERROR" in llm_context:
        return gr_state, llm_context, None

    system_prompt = (
        "You are a friendly, insightful AI Career Coach. "
        "Your purpose is to converse with the user about their career and potential job paths "
        "based on their resume.\n\n--- PROFILE CONTEXT ---\n"
        f"{llm_context}"
    )

    gr_state['system_prompt'] = system_prompt
    initial_message = "Welcome! I've analyzed your resume. How can I help guide your next steps today?"

    return gr_state, "✅ Analysis Complete! You can now chat about your career.", initial_message


def handle_chat(message, history, gr_state):
    """Handles chat interactions."""
    system_prompt = gr_state.get('system_prompt')
    if not system_prompt:
        return history + [[message, "Please upload and analyze your resume first."]]

    messages = [{"role": "system", "content": system_prompt}]
    for human, agent in history:
        messages.append({"role": "user", "content": human})
        messages.append({"role": "assistant", "content": agent})
    messages.append({"role": "user", "content": message})

    try:
        response = client.chat.completions.create(model=ollama_model, messages=messages)
        return history + [[message, response.choices[0].message.content]]
    except Exception as e:
        print(f"Error: {e}")
        return history + [[message, f"⚠️ Error communicating with LLM: {e}"]]


# --- 3. GRADIO INTERFACE ---

initial_state = {'system_prompt': None}

with gr.Blocks(title="Resume Coach Chatbot") as demo:
    gr.Markdown("## 📄 Resume Coach: Conversational Career Analysis (LLaMA 3.2)")

    app_state = gr.State(initial_state)

    with gr.Row():
        with gr.Column(scale=1):
            file_upload = gr.File(label="Upload Resume (PDF)", type="filepath")
            upload_button = gr.Button("Analyze Resume")
            status_output = gr.Textbox(label="Status", value="Upload your resume to begin.")

        with gr.Column(scale=3):
            chatbot = gr.Chatbot(label="AI Career Coach", height=450)
            chat_input = gr.Textbox(placeholder="Ask about your skills, job prospects, or career path...")

    upload_button.click(
        handle_file_upload,
        inputs=[file_upload, app_state],
        outputs=[app_state, status_output, chatbot],
        queue=False
    )

    chat_input.submit(
        handle_chat,
        inputs=[chat_input, chatbot, app_state],
        outputs=chatbot
    ).then(lambda: gr.update(value=''), outputs=[chat_input], queue=False)

if __name__ == "__main__":
    demo.launch()
