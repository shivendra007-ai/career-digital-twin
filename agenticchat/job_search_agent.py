import os
import requests
import logging
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RapidJobAgent:
    """
    An agent to aggregate job searches from multiple RapidAPI services
    using a single API Key.
    """
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers = {
            "X-RapidAPI-Key": self.api_key,
        }

    def _get_headers(self, host: str):
        """Helper to construct headers for specific hosts."""
        return {
            "X-RapidAPI-Key": self.api_key,
            "X-RapidAPI-Host": host
        }

    def search_jsearch(self, query: str, location: str = "Remote") -> List[Dict]:
        """
        Uses 'JSearch' API. This is often the most reliable general aggregator.
        """
        host = "jsearch.p.rapidapi.com"
        url = f"https://{host}/search"
        
        params = {
            "query": f"{query} in {location}",
            "page": "1",
            "num_pages": "1"
        }

        try:
            response = requests.get(url, headers=self._get_headers(host), params=params)
            response.raise_for_status()
            data = response.json()
            
            jobs = []
            if "data" in data:
                for item in data["data"]:
                    jobs.append({
                        "source": "JSearch",
                        "title": item.get("job_title", "Unknown Role"),
                        "company": item.get("employer_name", "Unknown Company"),
                        "location": item.get("job_city", location),
                        "link": item.get("job_apply_link", "#"),
                        "description": item.get("job_description", "")[:200] + "..."
                    })
            return jobs
        except Exception as e:
            logger.error(f"JSearch failed: {e}")
            return []

    def search_linkedin_scraper(self, query: str, location: str = "United States") -> List[Dict]:
        """
        Uses 'Real-Time LinkedIn Scraper API'.
        """
        host = "linkedin-data-api.p.rapidapi.com" 
        # Note: Endpoints for scrapers change often. This uses a common pattern.
        url = f"https://{host}/search-jobs"
        
        params = {
            "keywords": query,
            "locationId": "103644278", # Default generic ID (often US), or use location text if API supports it
            "datePosted": "anyTime",
            "sort": "mostRelevant"
        }

        try:
            response = requests.get(url, headers=self._get_headers(host), params=params)
            # Scrapers often return 200 even on empty data, so we check carefully
            if response.status_code == 200:
                data = response.json()
                jobs = []
                # Adjust parsing based on actual response structure of the specific scraper version
                items = data.get("data", []) if isinstance(data.get("data"), list) else []
                for item in items:
                    jobs.append({
                        "source": "LinkedIn",
                        "title": item.get("title", "Unknown"),
                        "company": item.get("company", {}).get("name", "Unknown"),
                        "location": item.get("location", location),
                        "link": item.get("url", "#"),
                        "description": "LinkedIn Listing"
                    })
                return jobs
            return []
        except Exception as e:
            logger.error(f"LinkedIn Search failed: {e}")
            return []

    def search_remote_jobs(self, query: str) -> List[Dict]:
        """
        Uses 'Remote Jobs' API.
        """
        host = "remote-jobs-api1.p.rapidapi.com"
        url = f"https://{host}/jobs/search"
        
        # This API typically uses POST
        payload = {
            "search": query,
            "country": "usa", # default
            "max_results": 10
        }
        
        headers = self._get_headers(host)
        headers["Content-Type"] = "application/json"

        try:
            response = requests.post(url, headers=headers, json=payload)
            if response.status_code == 200:
                data = response.json()
                jobs = []
                # Assuming standard list response
                results = data.get("results", []) if "results" in data else data
                if isinstance(results, list):
                    for item in results:
                        jobs.append({
                            "source": "RemoteJobs",
                            "title": item.get("title", "Remote Role"),
                            "company": item.get("company", "Unknown"),
                            "location": "Remote",
                            "link": item.get("url", "#"),
                            "description": "Remote Opportunity"
                        })
                return jobs
            return []
        except Exception as e:
            logger.error(f"Remote Jobs failed: {e}")
            return []

    def search_google_jobs(self, query: str) -> List[Dict]:
        """
        Uses 'Google Jobs API'.
        """
        host = "google-jobs-search.p.rapidapi.com"
        url = f"https://{host}/search"
        
        params = {"query": query}

        try:
            response = requests.get(url, headers=self._get_headers(host), params=params)
            if response.status_code == 200:
                data = response.json()
                jobs = []
                # Parsing logic depends on provider
                results = data.get("data", []) 
                for item in results:
                    jobs.append({
                        "source": "GoogleJobs",
                        "title": item.get("title", query),
                        "company": item.get("company_name", ""),
                        "location": item.get("location", ""),
                        "link": item.get("apply_link", "#"),
                        "description": "Google Jobs listing"
                    })
                return jobs
            return []
        except Exception as e:
            logger.error(f"Google Jobs failed: {e}")
            return []
            
    def search_internships(self, query: str) -> List[Dict]:
        """
        Uses 'Internships API'.
        """
        host = "internships-api.p.rapidapi.com"
        url = f"https://{host}/active-internships" # Example endpoint
        
        params = {"title": query}

        try:
            response = requests.get(url, headers=self._get_headers(host), params=params)
            if response.status_code == 200:
                data = response.json()
                jobs = []
                # Mock parsing logic - adjust based on real response
                if isinstance(data, list):
                    for item in data[:5]:
                        jobs.append({
                            "source": "Internships",
                            "title": item.get("title", f"{query} Intern"),
                            "company": item.get("company", ""),
                            "location": item.get("location", ""),
                            "link": item.get("url", "#"),
                            "description": "Internship opportunity"
                        })
                return jobs
            return []
        except Exception as e:
            logger.error(f"Internships Search failed: {e}")
            return []

    def broadcast_search(self, role_query: str, location: str = "Remote") -> List[Dict]:
        """
        Aggregates results from multiple APIs. 
        Prioritizes JSearch as it's often the most detailed.
        """
        all_jobs = []
        
        # 1. JSearch (Primary)
        print(f"Searching JSearch for {role_query}...")
        j_results = self.search_jsearch(role_query, location)
        all_jobs.extend(j_results)
        
        # 2. Remote Jobs (Secondary - if location is remote)
        if "remote" in location.lower():
            print(f"Searching Remote Jobs for {role_query}...")
            r_results = self.search_remote_jobs(role_query)
            all_jobs.extend(r_results)
            
        # 3. Google Jobs (Fallback)
        if len(all_jobs) < 5:
            print(f"Searching Google Jobs for {role_query}...")
            g_results = self.search_google_jobs(f"{role_query} in {location}")
            all_jobs.extend(g_results)

        # Deduplicate based on link or title+company
        unique_jobs = {job['link']: job for job in all_jobs}
        return list(unique_jobs.values())