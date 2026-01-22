import os
import smtplib
from email.message import EmailMessage
from dotenv import load_dotenv

# Load email credentials
load_dotenv()

EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASS = os.getenv("EMAIL_PASS")

def send_email(to_email: str, subject: str, body: str):
    """
    Connects to the email server and sends a message.
    """
    if not EMAIL_USER or not EMAIL_PASS:
        print("Email credentials (EMAIL_USER, EMAIL_PASS) not set in .env. Skipping email.")
        return False

    # Create the email message
    msg = EmailMessage()
    msg.set_content(body)
    msg["Subject"] = subject
    msg["From"] = EMAIL_USER
    msg["To"] = to_email

    try:
        # Connect to Gmail's SMTP server
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(EMAIL_USER, EMAIL_PASS)
            server.send_message(msg)
        print(f"Email successfully sent to {to_email}")
        return True
    except Exception as e:
        print(f"Error sending email: {e}")
        return False

def stub_job_search(job_title: str):
    """
    A *stub* function to simulate a job search.
    In a real app, this would use an API (like JSearch or SerpApi).
    """
    print(f"Simulating job search for: {job_title}")
    # Return 3 mock job descriptions
    return [
        {
            "title": "AI Agent Developer (Remote)",
            "company": "FutureTech Solutions",
            "description": "Seeking a developer skilled in Python, LangGraph, and multi-agent systems. You will build and deploy autonomous agents for our clients. Must have experience with Ollama and local LLMs."
        },
        {
            "title": "Data Scientist - LLM Research",
            "company": "InnovaCore AI",
            "description": "InnovaCore is looking for a Data Scientist to fine-tune and evaluate new LLMs. Requires deep knowledge of PyTorch, model quantization, and prompt engineering."
        },
        {
            "title": "Python Backend Engineer (AI Platforms)",
            "company": "ConnectSphere",
            "description": "We are hiring a backend engineer to build the infrastructure for our AI chatbots. Strong skills in Python, FastAPI, and database management are required. Experience with Streamlit is a plus."
        }
    ]