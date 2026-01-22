# lab3_ollama_auto.py
# Agentic loop (Generate -> Evaluate -> Rerun) converted to use only local Ollama 3.2
# Now includes automatic generation of summary.txt from linkedin.pdf
#python agenticchat\file3.py
# file3_advisor.py
# AI Career Advisor Agent with Generate -> Evaluate -> Rerun loop, using Ollama 3.2

import os
import json
from dotenv import load_dotenv
from openai import OpenAI
import gradio as gr
from pydantic import BaseModel

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

# --- PERSONA AND CONTEXT ---
name = "AI Career Advisor"


# --- 1. Define Prompts and Pydantic Model ---

# Pydantic model for structured evaluation output
class Evaluation(BaseModel):
    is_acceptable: bool
    feedback: str

# System Prompt for the Primary Agent (The AI Career Advisor)
system_prompt = f"You are acting as a highly experienced and creative **{name}**. " \
    f"Your purpose is to provide users with insightful, actionable, and futuristic career advice, " \
    f"focusing on emerging fields like Agentic AI, large language models, and digital transformation. " \
    f"Your tone should be **inspirational, sharp, and concise**, focusing on high-impact strategic guidance. " \
    f"You must use a compelling and unique opening line for your first message, and always maintain an encouraging yet rigorous style."

# --- CREATIVE LINE ADDED ---
# This instruction is added to guide the AI's initial output and persona
system_prompt += f"\n\n**MANDATORY OPENING:** Your first response to the user **MUST** be: " \
                 f"'Welcome, future innovator! The career landscape is shifting. Tell me about your current trajectory, and let's chart a course to where the real opportunities lie.'"

# System Prompt for the Evaluator Agent (using JSON structure)
evaluator_system_prompt = f"You are a Quality Control Evaluator for the **{name}**. " \
    f"Your task is to decide whether the Advisor's response is acceptable quality. " \
    f"The Advisor must be **inspirational, sharp, and concise**. Responses must provide strategic value. " \
    f"If the response is overly generic, lacks actionable advice, or is verbose, reject it. " \
    f"Evaluate the latest response, replying with a JSON object, and only a JSON object, that conforms to this Pydantic schema: {Evaluation.schema_json()}"


def evaluator_user_prompt(reply, message, history):
    user_prompt = f"Here's the conversation between the User and the Agent: \n\n{history}\n\n"
    user_prompt += f"Here's the latest message from the User: \n\n{message}\n\n"
    user_prompt += f"Here's the latest response from the Agent: \n\n{reply}\n\n"
    user_prompt += "Please evaluate the response based on the system prompt, replying with the requested JSON object, nothing else."
    return user_prompt

# --- 2. Agent Functions (Logic remains the same) ---

def evaluate(reply, message, history) -> Evaluation:
    """Evaluates a reply using the Ollama model with JSON parsing."""
    messages = [
        {"role": "system", "content": evaluator_system_prompt},
        {"role": "user", "content": evaluator_user_prompt(reply, message, history)}
    ]
    
    response = client.chat.completions.create(
        model=ollama_model, 
        messages=messages,
        response_format={"type": "json_object"} 
    )
    
    # Parse the JSON response manually
    try:
        json_output = response.choices[0].message.content
        data = json.loads(json_output)
        return Evaluation(is_acceptable=data.get("is_acceptable", False), feedback=data.get("feedback", "No feedback provided"))
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        print(f"Evaluation JSON parsing failed: {e}. Raw output: {response.choices[0].message.content[:100]}...")
        return Evaluation(is_acceptable=False, feedback=f"Failed to parse JSON response: {e}")


def rerun(reply, message, history, feedback):
    """Generates a new response after an initial reply is rejected."""
    print("--- RERUN ACTIVATED ---")
    updated_system_prompt = system_prompt + "\n\n## Previous answer rejected\nYou just tried to reply, but the quality control rejected your reply\n"
    updated_system_prompt += f"## Your attempted answer:\n{reply}\n\n"
    updated_system_prompt += f"## Reason for rejection:\\n{feedback}\\n\\n"
    
    messages = [
        {"role": "system", "content": updated_system_prompt}
    ] + history + [
        {"role": "user", "content": message}
    ]
    
    response = client.chat.completions.create(model=ollama_model, messages=messages)
    return response.choices[0].message.content


def chat(message, history):
    """Main chat handler with the Generate -> Evaluate -> Rerun loop."""
    # Gradio Compatibility Fix 
    history = [{"role": h["role"], "content": h["content"]} for h in history]

    # No "patent" logic needed for Career Advisor, so using base system prompt
    system = system_prompt
        
    messages = [
        {"role": "system", "content": system}
    ] + history + [
        {"role": "user", "content": message}
    ]
    
    # 1. GENERATE INITIAL REPLY
    response = client.chat.completions.create(model=ollama_model, messages=messages)
    reply = response.choices[0].message.content

    # 2. EVALUATE
    evaluation = evaluate(reply, message, history)
    
    # 3. RERUN IF FAILED
    if evaluation.is_acceptable:
        print("Passed evaluation - returning reply")
    else:
        print("Failed evaluation - retrying")
        print(f"Feedback: {evaluation.feedback}")
        reply = rerun(reply, message, history, evaluation.feedback)
        
    return reply

# --- 3. Launch Gradio Interface ---
# Launch the Gradio UI. The interface will open in your web browser.
print("\nLaunching Gradio Chat Interface...")
gr.ChatInterface(chat, type="messages").launch()