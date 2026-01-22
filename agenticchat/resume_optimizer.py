import json
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = "llama3.2"

class ATSScorer:
    def __init__(self):
        self.client = OpenAI(base_url=f"{OLLAMA_HOST}/v1", api_key="ollama")

    def score_resume(self, resume_text: str, job_description: str) -> dict:
        """
        Compares resume against job description and returns a score + missing keywords.
        """
        prompt = f"""
        Act as an Applicant Tracking System (ATS) algorithm.
        
        JOB DESCRIPTION:
        {job_description[:1500]}
        
        RESUME:
        {resume_text[:1500]}
        
        Task:
        1. Calculate a match percentage (0-100) based on skills and keywords.
        2. Identify 3-5 critical keywords missing from the resume.
        3. Provide one actionable tip to improve the resume for this specific job.
        
        Return ONLY a JSON object with keys: "match_score" (int), "missing_keywords" (list of strings), "improvement_tip" (string).
        """
        
        try:
            response = self.client.chat.completions.create(
                model=OLLAMA_MODEL,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content
            if "```json" in content:
                content = content.replace("```json", "").replace("```", "")
                
            return json.loads(content)
            
        except Exception as e:
            return {
                "match_score": 0, 
                "missing_keywords": ["Error analyzing"], 
                "improvement_tip": f"Analysis failed: {str(e)}"
            }