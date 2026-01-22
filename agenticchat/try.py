#streamlit run agenticchat/try.py
#streamlit run agenticchat/try2.py
#venv\Scripts\activate.bat

# =====================================
import os
import json
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader
from pydantic import BaseModel, Field
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
import tempfile
import uuid
import time
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
from pathlib import Path
import concurrent.futures # --- NEW: For Parallel Execution

# --- NEW IMPORT ---
# Ensure job_search_agent.py is in the same folder
try:
    from job_search_agent import RapidJobAgent
except ImportError:
    st.error("⚠️ 'job_search_agent.py' not found. Please ensure the file exists.")

# =====================================
# CONFIG SETUP (Updated for Robust File Handling)
# =====================================

# Use absolute path for robustness
CONFIG_FILE_PATH = Path(__file__).parent / "config.yaml"

def load_config():
    """Loads the YAML configuration file using the robust path."""
    if CONFIG_FILE_PATH.exists():
        with open(CONFIG_FILE_PATH, 'r') as f:
            # Use safe_load to prevent security issues and check for empty file
            content = f.read()
            if not content.strip():
                return create_default_config()
            return yaml.load(content, Loader=SafeLoader)
    else:
        return create_default_config()

def create_default_config():
    """Creates a default config object."""
    default = {
        "credentials": {"usernames": {}},
        "cookie": {
            "name": "career_twin_cookie",
            "key": str(uuid.uuid4()), 
            "expiry_days": 30
        }
    }
    # Attempt to write the default file immediately if missing
    try:
        with open(CONFIG_FILE_PATH, "w") as f:
            yaml.dump(default, f, default_flow_style=False)
        st.warning(f"Created a default 'config.yaml' at {CONFIG_FILE_PATH}. Please edit it with your hashes and secret key.")
    except Exception as e:
        st.error(f"Could not write default config. Please create config.yaml manually. Error: {e}")
    return default

def save_config(cfg):
    """Saves the current configuration back to the YAML file."""
    with open(CONFIG_FILE_PATH, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

# Load the config
config = load_config()


# =====================================
# AUTHENTICATOR (FIXED FOR OLDER VERSIONS)
# =====================================

try:
    authenticator = stauth.Authenticate(
        credentials=config["credentials"],
        cookie_name=config["cookie"]["name"],
        # ✅ FINAL FIX: Using the OLD parameter name 'cookie_key' 
        cookie_key=config["cookie"]["key"], 
        cookie_expiry_days=config["cookie"]["expiry_days"],
        # Removed 'preauthorized=[]'
    )
    AUTH_INITIALIZED = True

except Exception as e:
    st.error(f"Authentication initialization failed. Please check your library version or config.yaml content. Error: {e}")
    AUTH_INITIALIZED = False


# =====================================
# LLM INITIALIZATION
# =====================================

load_dotenv(override=True)
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = "llama3.2"

client = OpenAI(base_url=f"{OLLAMA_HOST}/v1", api_key="ollama")

st.set_page_config(
    page_title="Career Digital Twin",
    page_icon="🧠",
    layout="wide"
)


# =====================================
# SCHEMAS
# =====================================

class ResumeState(TypedDict):
    pdf_path: str
    resume_text: str
    analysis_result: dict
    job_listings: List[dict]
    error_message: str
    rapid_api_key: str # Pass key through state
    search_location: str # Pass location through state
    search_logs: List[str] # New field for debug logs

class JobMatchResult(BaseModel):
    best_job_titles: List[str]
    top_skills_matched: List[str]
    suggested_skills: List[str]
    career_path_summary: str

    def is_schema_output(self):
        return self.career_path_summary.strip().lower().startswith("a brief, motivational")


# =====================================
# HELPER FUNCTIONS
# =====================================

def generate_cover_letter(resume_text, job_title, company_name, job_desc):
    """Generates a cover letter using the LLM."""
    prompt = f"""
    You are a professional career coach. Write a compelling, concise cover letter for the user.
    
    JOB DETAILS:
    - Role: {job_title}
    - Company: {company_name}
    - Description: {job_desc}
    
    USER RESUME CONTEXT:
    {resume_text[:2000]} (truncated)
    
    The letter should be professional, enthusiastic, and highlight relevant matches from the resume.
    """
    try:
        response = client.chat.completions.create(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating cover letter: {e}"

def search_single_source(agent, source_name, title, location):
    """Helper to run a single search source safely."""
    try:
        if source_name == "JSearch":
            res = agent.search_jsearch(title, location=location)
        elif source_name == "LinkedIn":
            res = agent.search_linkedin_scraper(title, location=location)
        elif source_name == "GoogleJobs":
            query = f"{title} in {location}"
            res = agent.search_google_jobs(query)
        elif source_name == "RemoteJobs":
            res = agent.search_remote_jobs(title)
        else:
            return [], f"❓ Unknown source: {source_name}"
            
        if res:
            return res, f"✅ {source_name}: Found {len(res)} for '{title}'"
        else:
            return [], f"🔸 {source_name}: Found 0 for '{title}'"
            
    except Exception as e:
        return [], f"❌ {source_name} Error: {e}"


# =====================================
# LANGGRAPH NODES
# =====================================

def extract_resume_text(state: ResumeState) -> ResumeState:
    st.info("📄 Extracting text from your resume...")
    try:
        reader = PdfReader(state["pdf_path"])
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        if not text.strip():
            raise Exception("No extractable text found in resume.")
        return {
            "pdf_path": state["pdf_path"],
            "resume_text": text,
            "analysis_result": {},
            "job_listings": [],
            "error_message": "",
            "rapid_api_key": state["rapid_api_key"],
            "search_location": state["search_location"],
            "search_logs": []
        }
    except Exception as e:
        return {
            "pdf_path": state["pdf_path"],
            "resume_text": "",
            "analysis_result": {},
            "job_listings": [],
            "error_message": str(e),
            "rapid_api_key": state["rapid_api_key"],
            "search_location": state["search_location"],
            "search_logs": []
        }


def analyze_resume(state: ResumeState) -> ResumeState:
    if state["error_message"]:
        return state

    st.info("🤖 Analyzing resume using LLaMA 3.2...")

    system_prompt = (
        "Return ONLY a JSON object with:\n"
        "- best_job_titles (list of all relevant job titles found)\n"
        "- top_skills_matched (5 items)\n"
        "- suggested_skills (5 items)\n"
        "- career_path_summary (string)"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": state["resume_text"]}
    ]

    for _ in range(4):
        try:
            resp = client.chat.completions.create(
                model=OLLAMA_MODEL,
                messages=messages,
                response_format={"type": "json_object"}
            )

            raw = resp.choices[0].message.content.strip()
            # Clean potential markdown
            raw = raw.strip("```json").strip("```")

            parsed_dict = json.loads(raw)
            parsed = JobMatchResult.parse_obj(parsed_dict)

            if parsed.is_schema_output():
                raise Exception("Schema returned instead of actual data")

            return {
                "analysis_result": parsed.dict(),
                "resume_text": state["resume_text"],
                "error_message": "",
                "job_listings": [],
                "rapid_api_key": state["rapid_api_key"],
                "search_location": state["search_location"],
                "search_logs": []
            }

        except:
            time.sleep(1)

    return {
        "analysis_result": {},
        "resume_text": state["resume_text"],
        "error_message": "LLM failed to provide valid JSON.",
        "job_listings": [],
        "rapid_api_key": state["rapid_api_key"],
        "search_location": state["search_location"],
        "search_logs": []
    }

def search_jobs_node(state: ResumeState) -> ResumeState:
    """
    Takes the best job titles and searches RapidAPI in PARALLEL.
    """
    if state["error_message"] or not state["analysis_result"]:
        return state
    
    api_key = state.get("rapid_api_key")
    if not api_key:
        st.warning("⚠️ No RapidAPI Key provided. Skipping live job search.")
        return state

    location = state.get("search_location", "Remote")
    st.info(f"🔎 Agent is searching for live jobs in '{location}' via RapidAPI...")
    
    try:
        agent = RapidJobAgent(api_key)
    except NameError:
        return {**state, "error_message": "Job Agent not loaded. Check imports."}
    
    titles_to_search = state["analysis_result"].get("best_job_titles", [])
    
    if not titles_to_search:
        st.warning("⚠️ AI matched no job titles, so no search was performed.")
        return state

    all_jobs = []
    logs = []
    
    # --- PARALLEL EXECUTION SETUP ---
    sources = ["JSearch", "LinkedIn", "GoogleJobs", "RemoteJobs"]
    
    # Use ThreadPoolExecutor to run API calls concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_search = {}
        
        for title in titles_to_search:
            for source in sources:
                # Submit each search task to the pool
                future = executor.submit(search_single_source, agent, source, title, location)
                future_to_search[future] = f"{source} for {title}"
        
        # Process results as they complete
        progress_bar = st.progress(0)
        completed = 0
        total = len(future_to_search)
        
        for future in concurrent.futures.as_completed(future_to_search):
            jobs, log_msg = future.result()
            if jobs:
                all_jobs.extend(jobs)
            logs.append(log_msg)
            
            completed += 1
            progress_bar.progress(completed / total)
            
        progress_bar.empty()
    
    # Simple Deduplication
    unique_jobs = {}
    for job in all_jobs:
        key = job.get('link')
        if not key or key == "#":
            key = f"{job.get('title')}_{job.get('company')}"
        unique_jobs[key] = job

    final_jobs_list = list(unique_jobs.values())
    
    return {
        "analysis_result": state["analysis_result"],
        "resume_text": state["resume_text"],
        "error_message": "",
        "job_listings": final_jobs_list,
        "rapid_api_key": state["rapid_api_key"],
        "search_location": state["search_location"],
        "search_logs": logs
    }


# =====================================
# LANGGRAPH PIPELINE
# =====================================

workflow = StateGraph(ResumeState)
workflow.add_node("extract", extract_resume_text)
workflow.add_node("analyze", analyze_resume)
workflow.add_node("search_jobs", search_jobs_node)

workflow.set_entry_point("extract")
workflow.add_edge("extract", "analyze")
workflow.add_edge("analyze", "search_jobs")
workflow.add_edge("search_jobs", END)

analysis_pipeline = workflow.compile()


# =====================================
# MAIN APPLICATION
# =====================================

def run_app():
    # Initialize Session State
    if "analysis_result" not in st.session_state: st.session_state.analysis_result = {}
    if "job_listings" not in st.session_state: st.session_state.job_listings = []
    if "messages" not in st.session_state: st.session_state.messages = []
    if "analysis_done" not in st.session_state: st.session_state.analysis_done = False
    if "search_logs" not in st.session_state: st.session_state.search_logs = []
    # Store Resume Text in session for Cover Letter generation
    if "resume_text" not in st.session_state: st.session_state.resume_text = ""
    
    name = st.session_state.user_name
    
    # --- Sidebar for API Key ---
    with st.sidebar:
        st.divider()
        env_key = os.getenv("RAPIDAPI_KEY", "")
        if env_key:
            st.success("✅ RapidAPI Key Loaded")
            rapid_key = env_key
        else:
            st.warning("⚠️ Key not found in Environment")
            rapid_key = st.text_input("Enter RapidAPI Key", type="password")
        
        st.subheader("Job Search Settings")
        search_location = st.text_input("Job Location", value="Remote", help="e.g. 'New York', 'London', 'Remote'")

    st.header(f"👋 Welcome, {name}")
    st.markdown("Your personalized **Agentic Career Digital Twin** is ready. Upload your resume to begin.")

    # --- Initial Greeting ---
    if not st.session_state.messages or len(st.session_state.messages) < 2:
        st.session_state.messages = [] 
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"Hello, **{name}**! Upload your resume (PDF) and click 'Run Analysis' to begin."
        })


    # --- Resume Upload and Analysis Section ---
    uploaded = st.file_uploader("📎 Upload Resume (PDF)", type=["pdf"])
    
    if uploaded and st.button("🚀 Run Career Analysis"):
        temp_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}_{uploaded.name}")
        with open(temp_path, "wb") as f:
            f.write(uploaded.getbuffer())

        with st.spinner("Running agentic analysis pipeline (Extract -> Analyze -> Parallel Search)..."):
            result = analysis_pipeline.invoke(
                ResumeState(
                    pdf_path=temp_path, 
                    resume_text="", 
                    analysis_result={}, 
                    job_listings=[],
                    error_message="",
                    rapid_api_key=rapid_key,
                    search_location=search_location,
                    search_logs=[]
                )
            )
            
        if os.path.exists(temp_path):
             os.remove(temp_path)

        if result["error_message"]:
            st.error("❌ Analysis Failed: " + result["error_message"])
            st.session_state.analysis_done = False
            return

        st.session_state.analysis_result = result["analysis_result"]
        st.session_state.job_listings = result["job_listings"]
        st.session_state.search_logs = result.get("search_logs", [])
        st.session_state.resume_text = result.get("resume_text", "") # Save for later use
        st.session_state.analysis_done = True
        
        summary = result["analysis_result"]["career_path_summary"]
        job_count = len(result["job_listings"])

        if job_count > 0:
            st.success(f"✅ Analysis Completed. Found {job_count} relevant job openings!")
        else:
            st.warning("✅ Analysis Completed, but NO relevant jobs were found via API.")

        st.session_state.messages.append({
            "role": "assistant", 
            "content": f"✅ Analysis Complete! I've built your digital career twin, {name}. **Your summary is:** *{summary}*."
        })


        # --- Report Display ---
        output = result["analysis_result"]
        jobs = result["job_listings"]
        logs = st.session_state.search_logs

        st.subheader(f"🏆 Career Report for {name}")

        tab1, tab2, tab3 = st.tabs(["🎯 Job Match & Summary", "📈 Skills Breakdown", "💼 Live Job Search & Actions"])
        
        with tab1:
            st.subheader("Top Job Roles")
            for job in output.get("best_job_titles", []):
                st.markdown(f"- **{job}**")
            st.subheader("Career Path Summary")
            st.info(output.get("career_path_summary", "No summary available."))

        with tab2:
            st.subheader("Skills Overview")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Skills Matched:**")
                st.markdown("\n".join([f"✅ **{s}**" for s in output.get("top_skills_matched", [])]))
            with col2:
                st.markdown("**Skills to Learn Next:**")
                st.markdown("\n".join([f"🚀 **{s}**" for s in output.get("suggested_skills", [])]))

        with tab3:
            st.subheader(f"Live Opportunities in {search_location}")
            
            # --- Debug Expander ---
            with st.expander("🔍 Search Debug Log"):
                for log_item in logs:
                    if "❌" in log_item:
                        st.error(log_item)
                    elif "🔸" in log_item:
                        st.warning(log_item)
                    else:
                        st.success(log_item)

            if not jobs:
                st.info("No jobs found. Please check the Debug Log above.")
            else:
                for idx, job in enumerate(jobs):
                    with st.expander(f"{job['title']} @ {job['company']}"):
                        st.markdown(f"**Location:** {job['location']}")
                        st.markdown(f"**Source:** {job['source']}")
                        st.markdown(f"_{job['description'][:300]}..._")
                        
                        col_a, col_b = st.columns([1, 1])
                        
                        with col_a:
                            if job['link'] and job['link'] != "#":
                                st.markdown(f"[👉 Apply Here]({job['link']})")
                            else:
                                st.write("No direct link.")
                        
                        # --- Feature: Cover Letter Generation ---
                        with col_b:
                            if st.button(f"✍️ Draft Cover Letter", key=f"btn_{idx}"):
                                with st.spinner("Drafting cover letter..."):
                                    cl_text = generate_cover_letter(
                                        st.session_state.resume_text,
                                        job['title'],
                                        job['company'],
                                        job['description']
                                    )
                                    st.text_area("Generated Cover Letter:", value=cl_text, height=300)

    st.markdown("---")
    st.subheader("💬 Career Q&A")

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
    # Handle chat input
    if q := st.chat_input("Ask anything about your career..."):
        st.session_state.messages.append({"role": "user", "content": q})
        with st.chat_message("user"):
            st.markdown(q)
            
        with st.chat_message("assistant"):
            if not st.session_state.analysis_done:
                reply = "Please run the career analysis first to enable personalized coaching."
            else:
                jobs_context = ""
                if st.session_state.job_listings:
                    top_jobs = st.session_state.job_listings[:3]
                    jobs_context = f"\nAlso, here are 3 real job listings found for the user: {json.dumps(top_jobs)}"

                context = json.dumps(st.session_state.get("analysis_result", {}), indent=2) + jobs_context

                system_prompt = (
                    f"You are a friendly, insightful AI Career Coach speaking to {name}. "
                    "Use the following **CONTEXT** (resume analysis + real job listings) to answer concisely.\n"
                    f"{context}"
                )
                
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": q}
                ]

                try:
                    reply_obj = client.chat.completions.create(model=OLLAMA_MODEL, messages=messages)
                    reply = reply_obj.choices[0].message.content
                except Exception as e:
                    reply = f"⚠️ Error communicating with LLM during chat: {e}"

            st.markdown(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})


# =====================================
# 6. AUTH UI WRAPPER
# =====================================

if AUTH_INITIALIZED:

    st.title("🧠 Agentic Career Digital Twin")
    st.sidebar.header("Account")
    
    # Determine if we are logging in or registering
    mode = st.sidebar.radio("Mode", ["Login", "Register"])
    
    # --- LOGIN ---
    if mode == "Login":
        
        # --- FIX APPLIED: Removed 'form_name' ---
        authenticator.login(location="sidebar")
        
        # Check the status from session state after the call
        auth_status = st.session_state.get("authentication_status")

        if auth_status:
            # Successfully logged in
            st.session_state.user_name = st.session_state["name"]
            st.sidebar.success(f"Logged in as {st.session_state.user_name}!")
            authenticator.logout(button_name="Logout", location="sidebar")
            run_app()

        elif auth_status is False:
            st.sidebar.error("Incorrect username or password")

        elif auth_status is None:
            st.sidebar.info("Enter your credentials")
            st.info("Log in or register in the sidebar to access your personalized career coach.")

    # --- REGISTER ---
    else: # mode == "Register"
        
        st.subheader("Register New User")

        try:
            # --- FIX APPLIED: Removed 'form_name' ---
            email, username, name = authenticator.register_user(
                location="main"
            )

            if email:
                save_config(config)
                st.success("✅ Registration successful! Please login.")
                st.experimental_rerun() 

        except Exception as e:
            st.error("Registration failed: " + str(e))
            
else:
    st.title("🧠 Agentic Career Digital Twin (Setup Required)")
    st.error("Authentication system failed to initialize. Please check your `config.yaml` file.")