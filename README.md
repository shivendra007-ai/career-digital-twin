<div align="center">
  <h1>🧠 Career Digital Twin</h1>
  <p><i>An autonomous AI Career Agent designed to parse resumes, generate structured career paths, and track job market signals.</i></p>
  <p>Built with <b>LangGraph</b>, <b>Ollama (Llama 3.2)</b>, and <b>Streamlit</b>.</p>
</div>

---

## 📖 Overview
**Career Digital Twin** is an agentic AI system designed to automate and personalize your professional growth and job search process. Instead of manually tailoring resumes or researching roles, this application acts as a persistent "digital twin." It understands your specific background and leverages a local large language model to help you make faster, data-driven career decisions.

By utilizing **LangGraph** to orchestrate multi-step agentic workflows (chaining reasoning, tool calls, and memory) and **Ollama** for local inference, the system ensures your data remains private while delivering highly intelligent, contextual advice through an interactive **Streamlit** chat interface.

## ✨ Key Features
* **🤖 Agentic Reasoning Workflow:** Multi-step career decision-making and logical routing powered by LangGraph.
* **💬 Conversational Interface:** Natural, interactive chat UI built with Streamlit.
* **🧩 Local LLM Inference:** Powered entirely by Llama 3.2 running locally via Ollama—zero external API dependencies.
* **📄 Intelligent Resume Parsing:** Automatically extracts relevant skills, experiences, and educational background.
* **🗺️ Structured Career Paths:** Generates personalized, step-by-step progressions and milestones tailored to your professional goals.
* **⚙️ Job Automation:** Streamlines the discovery process and application strategies.

## 🛠️ Tech Stack

| Component | Technology |
| :--- | :--- |
| **Agent Orchestration** | [LangGraph](https://python.langchain.com/docs/langgraph) |
| **LLM Engine** | [Ollama](https://ollama.ai/) with Llama 3.2 |
| **Frontend** | [Streamlit](https://streamlit.io/) |
| **Core Language** | Python 3.10+ |

## 🚀 Getting Started

### Prerequisites
* **Python 3.10+**
* **Ollama** installed locally. Pull the Llama 3.2 model before running the application:
  ```bash
  ollama run llama3.2
  ```

### Installation

**1. Clone the repository:**
```bash
git clone https://github.com/shivendra007-ai/career-digital-twin.git
cd career-digital-twin
```

**2. Install dependencies:**
```bash
pip install -r requirements.txt
```

**3. Environment Setup:**
Create a `.env` file in the root directory to store any required local configurations or optional API keys.

### 💻 Usage

Launch the interactive frontend using Streamlit:
```bash
streamlit run agenticchat/app.py 
```

## 📂 Project Structure

```text
career-digital-twin/
├── agenticchat/        # Core LangGraph agent logic and Streamlit app
├── .env                # Environment variables (not committed)
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```

## 🗺️ Roadmap
- [ ] Advanced resume parsing and automated skill extraction.
- [ ] Job posting scraping and automated matching capabilities.
- [ ] Application tracking and analytics dashboard.
- [ ] Multi-agent workflow extension for deep company research and cover letter drafting.

## 👨‍💻 Author
**Shivendra Kushwaha**  
B.Tech Electronics and Communication Engineering (2023–2027)  
Jaypee Institute of Information Technology
