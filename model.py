import requests
import os

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyDez_YRD1X6659wsA9VPAeBUcb49vYQtOw")
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=" + GEMINI_API_KEY

PAST_JOBS = [
    {
        "title": "Full Stack Web Development for E-commerce Platform",
        "link": "https://www.upwork.com/job1",
        "review": "Excellent work, delivered on time and exceeded expectations!",
        "category": "Web Development"
    },
    {
        "title": "Logo and Brand Identity Design",
        "link": "https://www.upwork.com/job2",
        "review": "Creative and professional designer. Highly recommended!",
        "category": "Graphics Designing"
    },
]

PORTFOLIO = [
    {
        "name": "Modern Portfolio Website",
        "link": "https://www.behance.net/yourportfolio1",
        "category": "Web Development"
    },
    {
        "name": "Corporate Branding Kit",
        "link": "https://www.behance.net/yourportfolio2",
        "category": "Graphics Designing"
    },
]

def get_relevant_experience(job_category):
    jobs = [job for job in PAST_JOBS if job["category"].lower() in job_category.lower()]
    portfolio = [item for item in PORTFOLIO if item["category"].lower() in job_category.lower()]
    return jobs, portfolio

def generate_proposal(job_title, job_description, job_category="Web Development"):
    jobs, portfolio = get_relevant_experience(job_category)
    experience_text = "\n".join([
        f"- {job['title']} (Link: {job['link']})\n  Review: {job['review']}" for job in jobs
    ])
    portfolio_text = "\n".join([
        f"- {item['name']} (Link: {item['link']})" for item in portfolio
    ])
    prompt = f"""
        You are an expert freelancer on Upwork with 800+ completed jobs in Web Development and Graphics Designing. Write a professional, personalized proposal for the following job. Mention your relevant experience, include links to similar past jobs, and attach portfolio items. Use a friendly and confident tone.

        Job Title: {job_title}
        Job Description: {job_description}

        Relevant Experience:\n{experience_text}
        Portfolio:\n{portfolio_text}
        """
    headers = {"Content-Type": "application/json"}
    data = {
        "contents": [{"parts": [{"text": prompt}]}]
    }
    response = requests.post(GEMINI_API_URL, headers=headers, json=data)
    if response.status_code == 200:
        result = response.json()
        return result["candidates"][0]["content"]["parts"][0]["text"]
    else:
        return f"Error: {response.status_code} - {response.text}"
