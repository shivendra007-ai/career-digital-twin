# lab4_ollama_tools.py

#python agenticchat\file4.py
#http://127.0.0.1:7864
import os
import json
import requests
import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader

# --- CONFIGURATION & SETUP ---
load_dotenv(override=True)
ollama_host = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
ollama_model = "llama3.2"
name = "Ed Donner"

# Ollama Client configured for the local server
client = OpenAI(
    base_url=ollama_host + "/v1",
    api_key="ollama"
)
print(f"Ollama Client configured for: {ollama_host} using model: {ollama_model}")

# Pushover Configuration
pushover_user = os.getenv("PUSHOVER_USER")
pushover_token = os.getenv("PUSHOVER_TOKEN")
pushover_url = "https://api.pushover.net/1/messages.json"

if pushover_user:
    print(f"Pushover user found and starts with {pushover_user[0]}")
else:
    print("WARNING: Pushover user not found. Push notifications will be printed to console only.")


# --- TOOL DEFINITIONS: REAL-WORLD FUNCTIONS ---

def push(message):
    """Sends a push notification or prints to console if keys are missing."""
    print(f"Push Notification: {message}")
    if pushover_user and pushover_token:
        payload = {"user": pushover_user, "token": pushover_token, "message": message}
        try:
            requests.post(pushover_url, data=payload, timeout=5)
        except requests.exceptions.RequestException as e:
            print(f"ERROR sending Pushover notification: {e}")

def record_user_details(email: str, name: str = "Name not provided", notes: str = "not provided"):
    """Tool to record user interest and send a push notification."""
    message = f"Recording interest from {name} with email {email} and notes {notes}"
    push(message)
    return {"recorded": "ok", "message": "Interest successfully recorded and notification sent."}

def record_unknown_question(question: str):
    """Tool to record an unanswered question and send a push notification."""
    message = f"Recording unknown question: {question}"
    push(message)
    return {"recorded": "ok", "message": "Unknown question successfully recorded and notification sent."}


# --- TOOL DEFINITIONS: JSON SCHEMA ---

record_user_details_json = {
    "name": "record_user_details",
    "description": "Use this tool to record that a user is interested in being in touch and provided an email address",
    "parameters": {
        "type": "object",
        "properties": {
            "email": {"type": "string", "description": "The email address of this user"},
            "name": {"type": "string", "description": "The user's name, if they provided it"},
            "notes": {"type": "string", "description": "Any additional information about the conversation that's worth recording to give context"}
        },
        "required": ["email"],
        "additionalProperties": False
    }
}

record_unknown_question_json = {
    "name": "record_unknown_question",
    "description": "Always use this tool to record any question that couldn't be answered as you didn't know the answer",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {"type": "string", "description": "The question that couldn't be answered"},
        },
        "required": ["question"],
        "additionalProperties": False
    }
}

tools = [{"type": "function", "function": record_user_details_json},
         {"type": "function", "function": record_unknown_question_json}]


# --- TOOL CALL HANDLER (ELEGANT VERSION) ---

def handle_tool_calls(tool_calls):
    """Executes the functions requested by the LLM (Tool Calling)."""
    results = []
    for tool_call in tool_calls:
        tool_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)
        print(f"\nTool called: {tool_name} with args: {arguments}", flush=True)
        
        # Use globals() to call the function by name dynamically
        tool = globals().get(tool_name)
        result = tool(**arguments) if tool else {"error": f"Tool {tool_name} not found."}
        
        # Append the tool's result back to the messages list
        results.append({
            "role": "tool",
            "content": json.dumps(result),
            "tool_call_id": tool_call.id
        })
    return results


# --- DATA LOADING & AUTO-SUMMARY (FROM LAB 3) ---

def generate_summary_from_pdf(linkedin_text: str, summary_file_path: str, client: OpenAI) -> str:
    """Uses the LLM to generate a professional summary from the LinkedIn text and saves it."""
    print("--- Generating summary.txt automatically... ---")
    
    summarizer_prompt = f"You are a professional resume writer. Write a compelling, concise, " \
                        f"and professional summary for {name}, suitable for a personal website. " \
                        f"Write in the first person ('I', 'my') and base the summary only on the following text. " \
                        f"Respond with only the generated summary text, nothing else."
    
    messages = [
        {"role": "system", "content": summarizer_prompt},
        {"role": "user", "content": f"Here is the LinkedIn profile content:\n\n{linkedin_text}"}
    ]
    
    try:
        response = client.chat.completions.create(model=ollama_model, messages=messages)
        generated_summary = response.choices[0].message.content.strip()
        
        with open(summary_file_path, "w", encoding="utf-8") as f:
            f.write(generated_summary)
        
        print(f"✅ Summary saved to {summary_file_path}")
        return generated_summary
        
    except Exception as e:
        print(f"❌ Failed to generate summary with LLM: {e}. Using fallback.")
        return "Expert in Python, LLMs, and building reliable, scalable AI agents. (Fallback Summary)"


linkedin_path = "me/linkedin.pdf"
summary_path = "me/summary.txt"
linkedin = ""

try:
    # 1a. Load LinkedIn PDF text
    reader = PdfReader(linkedin_path)
    for page in reader.pages:
        text = page.extract_text()
        if text:
            linkedin += text
            
    if not linkedin.strip():
        raise ValueError("PDF extraction resulted in empty text.")
        
    print(f"✅ Successfully read content from {linkedin_path}.")

    # 1b. Load Summary or Generate it
    if os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = f.read()
        print(f"✅ Successfully loaded summary from {summary_path}.")
    else:
        summary = generate_summary_from_pdf(linkedin, summary_path, client)

except (FileNotFoundError, ValueError) as e:
    print(f"\n❌ WARNING: '{linkedin_path}' not found or empty. Using dummy text.")
    linkedin = "Experienced AI developer specializing in agentic systems."
    summary = "Expert in Python, LLMs, and building reliable, scalable AI agents. (Fallback Summary)"


# --- SYSTEM PROMPT (INCLUDING TOOL INSTRUCTIONS) ---

system_prompt = f"You are acting as {name}. You are answering questions on {name}'s website, " \
    f"particularly questions related to {name}'s career, background, skills and experience. " \
    f"Your responsibility is to represent {name} for interactions on the website as faithfully as possible. " \
    f"You are given a summary of {name}'s background and LinkedIn profile which you can use to answer questions. " \
    f"Be professional and engaging, as if talking to a potential client or future employer who came across the website. " \
    f"If you don't know the answer to any question, use your record_unknown_question tool to record the question that you couldn't answer, even if it's about something trivial or unrelated to career. " \
    f"If the user is engaging in discussion, try to steer them towards getting in touch via email; ask for their email and record it using your record_user_details tool. "

system_prompt += f"\n\n## Summary:\n{summary}\n\n## LinkedIn Profile:\n{linkedin}\n\n"
system_prompt += f"With this context, please chat with the user, always staying in character as {name}."


# --- MAIN CHAT FUNCTION WITH TOOL LOOP ---

def chat(message, history):
    """Main chat handler with the Tool Calling loop."""
    
    # Gradio Compatibility Fix for non-OpenAI APIs
    history = [{"role": h["role"], "content": h["content"]} for h in history]

    messages = [
        {"role": "system", "content": system_prompt}
    ] + history + [
        {"role": "user", "content": message}
    ]
    
    done = False
    while not done:
        
        # 1. CALL LLM with tools enabled (model is llama3.2)
        response = client.chat.completions.create(model=ollama_model, messages=messages, tools=tools)
        
        finish_reason = response.choices[0].finish_reason
        
        # 2. HANDLE TOOL CALLS
        if finish_reason == "tool_calls":
            tool_calls = response.choices[0].message.tool_calls
            
            # Execute the tool and get the results
            results = handle_tool_calls(tool_calls)
            
            # Append the LLM's request and the tool's results to the messages list for the next turn
            messages.append(response.choices[0].message)
            messages.extend(results)
        
        # 3. LLM finished with a text response
        else:
            done = True
            
    # Return the final message content
    return response.choices[0].message.content


# --- LAUNCH GRADIO INTERFACE ---
print("\nLaunching Gradio Chat Interface...")
gr.ChatInterface(chat, type="messages").launch()