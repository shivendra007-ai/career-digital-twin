# Career Digital Twin 🧠

An autonomous AI career agent that acts as your **digital twin** — analyzing your profile, tracking job market signals, and generating personalized career strategy, all through an interactive chat interface.

Built with **LangGraph**, **Ollama (Llama 3.2)**, and **Streamlit**.

---

## Overview

Career Digital Twin is an agentic AI system designed to automate and personalize the job search process. Instead of manually tailoring resumes, tracking applications, and researching roles, this agent acts as a persistent "digital twin" that understands your background and helps you make faster, more informed career decisions.

The system uses **LangGraph** to orchestrate multi-step agentic workflows — chaining together reasoning, tool calls, and memory — powered by a locally-run **Llama 3.2** model via **Ollama**, with a **Streamlit** front end for interaction.

---

## Features

- 🤖 **Agentic reasoning** — LangGraph-based workflow for multi-step career decision-making
- 💬 **Conversational interface** — Streamlit chat UI for natural interaction with your digital twin
- 🧩 **Local LLM inference** — runs on Llama 3.2 via Ollama, no external API dependency required
- 📄 **Career strategy generation** — personalized guidance based on your profile and goals
- ⚙️ **Job automation workflows** — streamlines repetitive parts of the job search process

*(Update this list with the specific capabilities your agent currently supports — e.g. resume parsing, job matching, application tracking, etc.)*

---

## Tech Stack

| Component | Technology |
|---|---|
| Agent orchestration | LangGraph |
| LLM | Llama 3.2 (via Ollama) |
| Frontend | Streamlit |
| Language | Python |

---

## Getting Started

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) installed locally, with the Llama 3.2 model pulled:
  ```bash
  ollama pull llama3.2
  ```

### Installation

1. Clone the repository
   ```bash
   git clone https://github.com/shivendra007-ai/career-digital-twin.git
   cd career-digital-twin
   ```

2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables
   Create a `.env` file in the root directory with the required keys (see `.env.example` if available).

4. Run the app
   ```bash
   streamlit run agenticchat/app.py
   ```
   *(Adjust the entry-point path to match your actual file structure)*

---

## Project Structure

```
career-digital-twin/
├── agenticchat/        # Core agent logic and Streamlit app
├── .env                # Environment variables (not committed)
├── requirements.txt    # Python dependencies
└── README.md
```

---

## Roadmap

- [ ] Resume parsing and skill extraction
- [ ] Job posting scraping and matching
- [ ] Application tracking dashboard
- [ ] Multi-agent workflow for research + drafting + review



---

## Author

**Shivendra**
B.Tech ECE, Jaypee Institute of Information Technology, Noida

