Career Digital Twin
An autonomous AI Career Agent designed to parse resumes and generate structured career paths. This project provides personalized career strategy and job automation through an intelligent, agentic workflow.

Overview
The Career Digital Twin is an agentic AI application that leverages local large language models to provide actionable, data-driven career guidance. By analyzing user inputs and generating customized strategies, it serves as a personal advisor for navigating professional growth.

Features
Resume Parsing: Intelligently extracts relevant skills, experiences, and educational background from resumes.

Structured Career Paths: Generates step-by-step progressions and milestones tailored to individual professional goals.

Personalized Strategy: Delivers customized advice and feedback based on the user's specific profile and target industry.

Job Automation: Streamlines the job discovery process and application strategies.

Tech Stack
LangGraph: Orchestrates the multi-agent workflow and logical routing.

Ollama (Llama 3.2): Serves as the local large language model powering the AI reasoning.

Streamlit: Provides an interactive, user-friendly frontend interface.

Python: Core programming language.

Getting Started
Prerequisites
Python 3.8+

Ollama installed locally.

Ensure the Llama 3.2 model is pulled via Ollama:

Bash
ollama run llama3.2
Installation
Clone the repository:

Bash
git clone https://github.com/shivendra007-ai/career-digital-twin.git
cd career-digital-twin
Install the required dependencies:

Bash
pip install -r requirements.txt
Configure your environment variables by adding the necessary API keys or local configurations to your .env file.

Usage
Launch the application using Streamlit:

Bash
streamlit run agenticchat/app.py
