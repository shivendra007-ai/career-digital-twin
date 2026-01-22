import json
from openai import OpenAI
import os
from dotenv import load_dotenv

# Load env to get host if needed, though we pass client usually
load_dotenv()
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = "llama3.2"

class InterviewCoach:
    def __init__(self):
        self.client = OpenAI(base_url=f"{OLLAMA_HOST}/v1", api_key="ollama")

    def generate_questions(self, job_title: str, company: str, description: str, resume_context: str) -> list:
        """
        Generates 3 targeted interview questions (1 Technical, 1 Behavioral, 1 Situational).
        """
        prompt = f"""
        You are an expert technical interviewer for {company}.
        
        JOB CONTEXT:
        Role: {job_title}
        Description: {description[:1000]}
        
        CANDIDATE CONTEXT:
        {resume_context[:1000]}
        
        Generate exactly 3 interview questions for this candidate:
        1. One specific technical question based on the job description.
        2. One behavioral question based on their resume gaps or strengths.
        3. One situational question (e.g., "Tell me about a time...").
        
        Return ONLY a JSON array of strings. Example: ["Question 1", "Question 2", "Question 3"]
        """
        
        try:
            response = self.client.chat.completions.create(
                model=OLLAMA_MODEL,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content
            # Handle potential JSON wrapping
            if "```json" in content:
                content = content.replace("```json", "").replace("```", "")
            
            data = json.loads(content)
            # Support both list directly or key "questions"
            if isinstance(data, list):
                return data
            return data.get("questions", ["Tell me about yourself.", "Why this role?", "What are your strengths?"])
            
        except Exception as e:
            return [f"Could not generate specific questions. Error: {str(e)}"]

    def evaluate_answer(self, question: str, user_answer: str) -> str:
        """
        Provides feedback on a user's answer.
        """
        prompt = f"""
        You are an interview coach. 
        
        Question: "{question}"
        Candidate Answer: "{user_answer}"
        
        Provide a concise critique:
        1. What was good?
        2. What was missing?
        3. A better way to phrase it (STAR method).
        """
        
        response = self.client.chat.completions.create(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content