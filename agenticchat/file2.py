# lab2_ollama.py
# This script converts the multi-provider LLM comparison to use only the local Ollama server (llama3.2)

import os
import json
from dotenv import load_dotenv
from openai import OpenAI
# The original notebook imported Anthropic, but we will use the OpenAI client for Ollama

# --- 1. Setup ---
# Always remember to do this!
load_dotenv(override=True)

# Ollama Setup: Get the host from environment variables or use the default.
ollama_host = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
ollama_model = "llama3.2" # Using your specific model

print(f"Ollama Host/API Base set to: {ollama_host}")
print(f"Using local model: {ollama_model}")
print("Ensure Ollama is running and you have pulled this model.")
print("-" * 30)

# Create the client instance pointing to the local Ollama server
client = OpenAI(
    base_url=ollama_host + '/v1',
    api_key='ollama'
)

# --- 2. Generate the Question ---
request = "Please come up with a challenging, nuanced question that I can ask a number of LLMs to evaluate their intelligence. "
request += "Answer only with the question, no explanation."
messages = [{"role": "user", "content": request}]

# Use Ollama to generate the question
print("Generating evaluation question using Ollama...")
response = client.chat.completions.create(
    model=ollama_model,
    messages=messages,
)
question = response.choices[0].message.content
print("Question Generated: ", question)
print("-" * 30)


# --- 3. Set up the Competitors and Answers ---
# In this simplified version, we treat the SAME model instance as 5 "competitors" 
# to simulate the original notebook's structure and run the judging logic.
competitors = []
answers = []
# The messages list for the competition responses is now based on the generated question
competition_messages = [
    {"role": "user", "content": question}
]

# --- 4. Run the Competitors (Replicating the original structure with one model) ---

# Competitor 1: Simulating gpt-4o-mini
model_name = "Ollama-llama3.2 (C1-Base)"
response = client.chat.completions.create(model=ollama_model, messages=competition_messages)
answer = response.choices[0].message.content
print(f"Response from {model_name} captured.")
competitors.append(model_name)
answers.append(answer)

# Competitor 2: Simulating claude-3-7-sonnet-latest
model_name = "Ollama-llama3.2 (C2-Claude Sim)"
response = client.chat.completions.create(model=ollama_model, messages=competition_messages)
answer = response.choices[0].message.content
print(f"Response from {model_name} captured.")
competitors.append(model_name)
answers.append(answer)

# Competitor 3: Simulating gemini-2.0-flash
model_name = "Ollama-llama3.2 (C3-Gemini Sim)"
response = client.chat.completions.create(model=ollama_model, messages=competition_messages)
answer = response.choices[0].message.content
print(f"Response from {model_name} captured.")
competitors.append(model_name)
answers.append(answer)

# Competitor 4: Simulating deepseek-chat
model_name = "Ollama-llama3.2 (C4-Deepseek Sim)"
response = client.chat.completions.create(model=ollama_model, messages=competition_messages)
answer = response.choices[0].message.content
print(f"Response from {model_name} captured.")
competitors.append(model_name)
answers.append(answer)

# Competitor 5: Simulating llama-3.3-70b-versatile (from Groq)
model_name = "Ollama-llama3.2 (C5-Groq Sim)"
response = client.chat.completions.create(model=ollama_model, messages=competition_messages)
answer = response.choices[0].message.content
print(f"Response from {model_name} captured.")
competitors.append(model_name)
answers.append(answer)

# Competitor 6: The explicit Ollama cell from the original notebook
model_name = "Ollama-llama3.2 (C6-Ollama Explicit)"
response = client.chat.completions.create(model=ollama_model, messages=competition_messages)
answer = response.choices[0].message.content
print(f"Response from {model_name} captured.")
competitors.append(model_name)
answers.append(answer)

print("-" * 30)

# --- 5. Combine Answers for the Judge ---

# Combine answers using the original notebook's logic
together = ""
for index, answer in enumerate(answers):
    together += f"# Response from competitor {index+1}\n\n"
    together += answer + "\n\n"

# The prompt for the Judge model
judge_prompt = f"""You are judging a competition between {len(competitors)} competitors.
Each model has been given this question:

{question}

Your job is to evaluate each response for clarity and strength of argument, and rank them in order of best to worst.
Respond with JSON, and only JSON, with the following format:
{{"results": ["best competitor number", "second best competitor number", "third best competitor number", ...]}}

Here are the responses from each competitor:

{together}

Now respond with the JSON with the ranked order of the competitors, nothing else. Do not include markdown formatting or code blocks."""

print("Judge Prompt created. Sending for evaluation...")
print("-" * 30)


# --- 6. Judgement Time ---

judge_messages = [{"role": "user", "content": judge_prompt}]

# Use Ollama as the judge (since all others are disabled)
# Using llama3.2 for judging since it's the only configured model
response = client.chat.completions.create(
    model=ollama_model,
    messages=judge_messages,
)
results = response.choices[0].message.content

print("Raw Judge Results (JSON):")
print(results)
print("-" * 30)


# --- 7. Print Final Ranks ---

try:
    results_dict = json.loads(results)
    ranks = results_dict["results"]
    print("Final Rankings:")
    for index, result in enumerate(ranks):
        # Convert result to integer and access competitors list (result-1 is the correct index)
        competitor = competitors[int(result)-1]
        print(f"Rank {index+1}: {competitor}")
except (json.JSONDecodeError, IndexError, ValueError) as e:
    print(f"Error processing judge's JSON response: {e}")
    print("The model might not have responded with perfect JSON. Review the raw results.")