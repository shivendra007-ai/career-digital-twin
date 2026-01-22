import json
import os
from datetime import datetime

MEMORY_FILE = "memory.json"

class MemoryManager:
    def __init__(self):
        # Initialize file if it doesn't exist
        if not os.path.exists(MEMORY_FILE):
            with open(MEMORY_FILE, "w") as f:
                json.dump({
                    "job_search_history": [],
                    "top_roles_history": [],
                    "interview_scores": [],
                    "resume_uploads": [],
                    "chat_history": [] # <--- Ensure this field exists
                }, f, indent=4)
        else:
            # Ensure new fields exist in old files (migration logic)
            data = self.load()
            if "chat_history" not in data:
                data["chat_history"] = []
                self.save(data)

    def load(self):
        with open(MEMORY_FILE, "r") as f:
            return json.load(f)

    def save(self, data):
        with open(MEMORY_FILE, "w") as f:
            json.dump(data, f, indent=4)

    # --- NEW METHODS FOR CHAT (Fixes AttributeError) ---
    def save_chat_history(self, messages):
        """Saves the full list of chat messages."""
        data = self.load()
        data["chat_history"] = messages
        self.save(data)

    def get_chat_history(self):
        """Retrieves the full list of chat messages."""
        data = self.load()
        return data.get("chat_history", [])

    def clear_chat_history(self):
        """Clears the chat history from memory."""
        data = self.load()
        data["chat_history"] = []
        self.save(data)

    # --- EXISTING METHODS ---
    def add_job_search(self, role, location, results_count):
        data = self.load()
        data["job_search_history"].append({
            "role": role,
            "location": location,
            "results": results_count,
            "timestamp": datetime.now().isoformat()
        })
        self.save(data)

    def add_roles(self, roles_list):
        data = self.load()
        data["top_roles_history"].append({
            "roles": roles_list,
            "timestamp": datetime.now().isoformat()
        })
        self.save(data)

    def add_interview_score(self, role, score, feedback):
        data = self.load()
        data["interview_scores"].append({
            "role": role,
            "score": score,
            "feedback": feedback,
            "timestamp": datetime.now().isoformat()
        })
        self.save(data)

    def add_resume_upload(self, extracted_skills):
        data = self.load()
        data["resume_uploads"].append({
            "skills": extracted_skills,
            "timestamp": datetime.now().isoformat()
        })
        self.save(data)