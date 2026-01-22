# First let's do an import. If you get an Import Error, double check that your Kernel is correct..
from dotenv import load_dotenv

# Next it's time to load the environment variables
# This is important for loading the OLLAMA_HOST from your .env file (if you set it).
load_dotenv(override=True)

# Ollama Setup: Get the host from environment variables or use the default.
# Note: Ollama doesn't require an API key.
import os
ollama_host = os.getenv('OLLAMA_HOST', 'http://localhost:11434')

print(f"Ollama Host/API Base set to: {ollama_host}\n")
print("Ensure Ollama is running and you have pulled a model like 'llama3' with 'ollama pull llama3'")

# Import the OpenAI client, which is used to communicate with Ollama's API endpoint.
from openai import OpenAI

# Create an instance of the OpenAI client, redirecting it to the Ollama server.
# We specify the 'base_url' and use a dummy 'api_key'.
client = OpenAI(
    base_url = ollama_host + "/v1",
    api_key = "ollama_key"
)

# Create a list of messages in the standard format.
# (Syntax corrected: removed unnecessary backslashes)
messages = [{"role": "user", "content": "What is 2+2?"}]

# Call the API using the local Ollama model (e.g., 'llama3').
response = client.chat.completions.create(
    model="llama3.2",
    messages=messages
)

print("\n--- Initial Math Query ---")
print(response.choices[0].message.content)

# Prepare a new question.
question = "Please propose a hard, challenging question to assess someone's IQ. Respond only with the question."
messages = [{"role": "user", "content": question}]

# Ask it again using the Ollama model ('llama3').
response = client.chat.completions.create(
    model="llama3.2",
    messages=messages
)

question = response.choices[0].message.content
print("\n--- Generated IQ Question ---")
print(question)

# Form a new messages list with the question.
messages = [{"role": "user", "content": question}]

# Ask for the answer using the Ollama model ('llama3').
response = client.chat.completions.create(
    model="llama3.2",
    messages=messages
)

answer = response.choices[0].message.content
print("\n--- Answer to IQ Question ---")
print(answer)

# The following lines are for display in a Notebook environment, but are kept for completeness.
# If you run this in a standard terminal, they will cause an ImportError and should be commented out.
# from IPython.display import Markdown, display
# display(Markdown(answer))


# --- Exercise Block ---

print("\n--- Starting Exercise Block ---")

# First create the messages:
# (Syntax corrected: removed unnecessary backslashes)
messages = [{"role": "user", "content": "Propose a unique business area for an Agentic AI solution."}]

# Then make the first call (using 'llama3'):
response = client.chat.completions.create(
    model="llama3.2",
    messages=messages
)

# Then read the business idea:
business_idea = response.choices[0].message.content
print(f"Business Idea: {business_idea[:100]}...")

# And repeat! In the next message, include the business idea within the message
messages = [{"role": "user", "content": f"Given the business area: {business_idea}, what is a major pain-point or challenge ripe for an Agentic solution? Respond only with the pain point."}]

response = client.chat.completions.create(
    model="llama3.2",
    messages=messages
)

pain_point = response.choices[0].message.content
print(f"Pain Point: {pain_point[:100]}...")