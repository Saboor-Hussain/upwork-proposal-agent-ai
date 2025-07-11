import os
import pickle
import numpy as np
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType, Tool
from langchain.callbacks import get_openai_callback
from langchain.schema import HumanMessage
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import csv


load_dotenv()
EMBEDDINGS_FILE = './embeddings/website_embeddings_1.pkl'
UPWORK_EMBEDDINGS_FILE = './embeddings/upwork_embeddings.pkl'
FIGMA_EMBEDDINGS_FILE = './embeddings/figma_embeddings.pkl'
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

def load_embeddings():
    with open(EMBEDDINGS_FILE, 'rb') as f:
        data = pickle.load(f)
    return data['embeddings'], data['websites']

def load_upwork_embeddings():
    with open(UPWORK_EMBEDDINGS_FILE, 'rb') as f:
        data = pickle.load(f)
    return data['embeddings'], data['websites']

def load_figma_embeddings():
    with open(FIGMA_EMBEDDINGS_FILE, 'rb') as f:
        data = pickle.load(f)
    return data['embeddings'], data['figmas']

def get_relevant_portfolio(job_desc, top_k=6):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings, websites = load_embeddings()
    job_emb = model.encode([job_desc])[0]
    similarities = np.dot(embeddings, job_emb) / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(job_emb))
    top_indices = similarities.argsort()[-top_k:][::-1]
    # Filter for strong relevance: only include if similarity is above a threshold (e.g., 0.5)
    relevant = [(websites[i], similarities[i]) for i in top_indices if similarities[i] > 0.5]
    if len(relevant) < top_k:
        for i in top_indices:
            if (websites[i], similarities[i]) not in relevant:
                relevant.append((websites[i], similarities[i]))
            if len(relevant) == top_k:
                break
    return [w for w, _ in relevant[:top_k]]


def get_relevant_portfolio_upwork(job_desc, top_k=3):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings, websites = load_upwork_embeddings()
    job_emb = model.encode([job_desc])[0]
    similarities = np.dot(embeddings, job_emb) / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(job_emb))
    top_indices = similarities.argsort()[-top_k:][::-1]
    relevant = [(websites[i], similarities[i]) for i in top_indices if similarities[i] > 0.5]
    if len(relevant) < top_k:
        for i in top_indices:
            if (websites[i], similarities[i]) not in relevant:
                relevant.append((websites[i], similarities[i]))
            if len(relevant) == top_k:
                break
    return [w for w, _ in relevant[:top_k]]


def get_relevant_portfolio_figma(job_desc, top_k=3):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings, figmas = load_figma_embeddings()
    job_emb = model.encode([job_desc])[0]
    similarities = np.dot(embeddings, job_emb) / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(job_emb))
    top_indices = similarities.argsort()[-top_k:][::-1]
    relevant = [(figmas[i], similarities[i]) for i in top_indices if similarities[i] > 0.5]
    if len(relevant) < top_k:
        for i in top_indices:
            if (figmas[i], similarities[i]) not in relevant:
                relevant.append((figmas[i], similarities[i]))
            if len(relevant) == top_k:
                break
    return [w for w, _ in relevant[:top_k]]



def portfolio_tool_func(job_desc: str):
    """Tool to retrieve relevant portfolio links for a job description."""
    portfolio = get_relevant_portfolio(job_desc)
    return '\n'.join([f"- {w['Website Links']} ({w['Category']})" for w in portfolio])


def upwork_portfolio_tool_func(job_desc: str):
    """Tool to retrieve relevant Upwork portfolio links for a job description."""
    portfolio = get_relevant_portfolio_upwork(job_desc)
    return '\n'.join([f"- {w['Website Link']} ({w['Category']})" for w in portfolio])

def figma_portfolio_tool_func(job_desc: str):
    """Tool to retrieve relevant Figma portfolio links for a job description."""
    portfolio = get_relevant_portfolio_figma(job_desc)
    return '\n'.join([f"- {w['Figma Link']} ({w['Category']})" for w in portfolio])


portfolio_tool = Tool(
    name="Portfolio Retriever",
    func=portfolio_tool_func,
    description="Given a job description, returns 2-3 relevant portfolio links from the matching niche."
)

upwork_portfolio_tool = Tool(
    name="Upwork Portfolio Retriever",
    func=upwork_portfolio_tool_func,
    description="Given a job description, returns 1-2 relevant Upwork Case Study links from the matching niche."
)

figma_portfolio_tool = Tool(
    name="Figma Portfolio Retriever",
    func=figma_portfolio_tool_func,
    description="Given a job description, returns 1-2 relevant Figma Portfolio links from the matching niche."
)


def main():
    llm = ChatOpenAI(model="gpt-4.1-mini", api_key=OPENAI_API_KEY)

    agent = initialize_agent(
        tools=[portfolio_tool, upwork_portfolio_tool],
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=False
    )
    # job_title = "Coaching Website for Online Wellness Program"
    # job_desc = """
    #         I’m a wellness coach launching an \
    #         online program. Need a landing site with booking calendar, testimonials, video content, blog, and \
    #         integration with Teachable or Kajabi.
    #     """

    job_title = str(input("Enter job title: "))
    job_desc = str(input("Enter job description: "))

    techs = []
    delivery_estimate = "15 days"
    user_prompt = f"""
        You are Muhammad Ibrahim, an expert Upwork freelancer and proposal writer agent.
        Write a personalized Upwork proposal for the following job. Follow these rules:
        1. Start with a personalized greeting.
        2. Mention the job/niche context in the first line.
        3. Use the Portfolio Retriever tool to include 2–3 relevant portfolio links ONLY from the matching niche.
        4. Use the Upwork Portfolio Retriever tool to include 1–2 relevant Upwork Case Study links ONLY from the matching niche.
        5. If the job description includes "Figma" or "figma", use the Figma Portfolio Retriever tool. Mention, "This is the website we have designed and the prototype link is [prototype link]. This is what we have developed [website link]." Ensure the project name matches.
        6. If the job description does not include "Figma" or "figma", use the Figma Portfolio Retriever tool to include the Figma link of the relevant category design prototype.
        7. Follow the Creasions writing style (calm, confident, helpful).
        8. Mention tech/tools if listed in the job: {', '.join(techs)}
        9. Give timeline or delivery estimate if possible: {delivery_estimate}
        10. End with a CTA (e.g., let’s connect, would love to chat).

        Job Title: {job_title}
        Job Description: {job_desc}
    """


    with get_openai_callback() as cb:
        result = agent.run(user_prompt)
        print(result)

        # Token usage stats
        print("\n\n","="*40,"Token Usage Stats","="*40)
        print(f"\nTotal tokens used: {cb.total_tokens}")
        print(f"Prompt tokens: {cb.prompt_tokens}")
        print(f"Completion tokens: {cb.completion_tokens}")
        print(f"Total cost (USD): ${cb.total_cost}")
        print("="*120)

def evaluate_proposal(agent, job_title, job_desc, expected_output, techs=None, delivery_estimate="15 days"): 
    techs = techs or []
    user_prompt = f"""
        You are Muhammad Ibrahim, an expert Upwork freelancer and proposal writer agent.
        Write a personalized Upwork proposal for the following job. Follow these rules:
        1. Start with a personalized greeting.
        2. Mention the job/niche context in the first line.
        3. Use the Portfolio Retriever tool to include 2–3 relevant portfolio links ONLY from the matching niche.
        4. Use the Upwork Portfolio Retriever tool to include 1–2 relevant Upwork Case Study links ONLY from the matching niche.
        5. If the job description includes "Figma" or "figma", use the Figma Portfolio Retriever tool. Mention, "This is the website we have designed and the prototype link is [prototype link]. This is what we have developed [website link]." Ensure the project name matches.
        6. If the job description does not include "Figma" or "figma", use the Figma Portfolio Retriever tool to include the Figma link of the relevant category design prototype.
        7. Follow the Creasions writing style (calm, confident, helpful).
        8. Mention tech/tools if listed in the job: {', '.join(techs)}
        9. Give timeline or delivery estimate if possible: {delivery_estimate}
        10. End with a CTA (e.g., let’s connect, would love to chat).

        Job Title: {job_title}
        Job Description: {job_desc}
    """

    
    with get_openai_callback() as cb:
        actual_output = agent.run(user_prompt)
    print("\nExpected Output:\n", expected_output)
    print("\nActual Output:\n", actual_output)

    # Use LLM to evaluate similarity/quality
    llm = ChatOpenAI(model="gpt-4.1-mini", api_key=OPENAI_API_KEY)
    eval_prompt = f"""
    Compare the following two Upwork proposals for the same job. Give a similarity score from 0 to 1 (1 = identical, 0 = completely different) and a brief justification.\n\nExpected Proposal:\n{expected_output}\n\nActual Proposal:\n{actual_output}\n\nScore and justification:"""
    eval_result = llm([HumanMessage(content=eval_prompt)]).content
    print("\nEvaluation Result:\n", eval_result)
    return actual_output, eval_result

def evaluate_website_retrieval(job_desc, expected_website_link=None):
    """
    Evaluate the top predicted website for a job description using the portfolio retrieval tool.
    """
    portfolio = get_relevant_portfolio(job_desc, top_k=1)
    predicted_website = portfolio[0]['Website Links'] if portfolio else None
    print(f"Job: {job_desc}")
    print(f"Accurate Website: {expected_website_link}")
    print(f"Predicted Website: {predicted_website}")
    print()
    return predicted_website

def generate_llm_jobs(n=5):
    """
    Generate n synthetic job titles and descriptions using the LLM.
    Returns a list of dicts: [{"title": ..., "description": ...}, ...]
    """
    llm = ChatOpenAI(model="gpt-4.1-nano", api_key=OPENAI_API_KEY)
    prompt = f"""
    Generate {n} realistic Upwork job postings for web development, each with a job title and a detailed job description. Return as a JSON list of objects with 'title' and 'description' fields. Example: [{{'title': '...', 'description': '...'}}, ...]
    """
    response = llm([HumanMessage(content=prompt)]).content
    import json
    try:
        jobs = json.loads(response)
    except Exception:
        import re
        match = re.search(r'\[.*\]', response, re.DOTALL)
        if match:
            jobs = json.loads(match.group(0))
        else:
            jobs = []
    return jobs

def generate_jobs_from_websites(n=5):
    """
    For n random websites from the embeddings, generate a job description using the LLM.
    Returns a list of dicts: [{"job_desc": ..., "accurate_website": ...}, ...]
    """
    import random
    _, websites = load_embeddings()
    if n > len(websites):
        n = len(websites)
    selected = random.sample(websites, n)
    llm = ChatOpenAI(model="gpt-4.1-nano", api_key=OPENAI_API_KEY)
    jobs = []
    for w in selected:
        meta = f"Website: {w['Website Links']}\nCategory: {w['Category']}\nDescription: {w['Description']}\nKeywords: {w['Keywords']}\nTechstacks: {w['Techstacks']}"
        prompt = f"""
        Given the following website portfolio entry, write a realistic Upwork job description for a client who would need a website like this. Only output the job description, not the website link.
        {meta}
        """
        job_desc = llm([HumanMessage(content=prompt)]).content.strip()
        jobs.append({"job_desc": job_desc, "accurate_website": w['Website Links']})
    return jobs

def generate_jobs_from_website_combinations(n=5, k=3):
    """
    For n jobs, randomly select k websites with the same category, and generate a job description using their combined metadata.
    Returns a list of dicts: [{"job_desc": ..., "accurate_websites": [...]}, ...]
    """
    import random
    _, websites = load_embeddings()
    # Group websites by category
    from collections import defaultdict
    cat_map = defaultdict(list)
    for w in websites:
        cat_map[w['Category']].append(w)
    jobs = []
    llm = ChatOpenAI(model="gpt-4.1-mini", api_key=OPENAI_API_KEY)
    categories = [cat for cat in cat_map if len(cat_map[cat]) >= k]
    for _ in range(n):
        cat = random.choice(categories)
        selected = random.sample(cat_map[cat], k)
        meta = '\n\n'.join([
            f"Website: {w['Website Links']}\nDescription: {w['Description']}\nKeywords: {w['Keywords']}\nTechstacks: {w['Techstacks']}"
            for w in selected
        ])
        prompt = f"""
        Given the following {k} website portfolio entries (all from the same category), write a realistic Upwork job description for a client who would need a website with these combined features and requirements. Only output the job description, not the website links.
        {meta}
        """
        job_desc = llm([HumanMessage(content=prompt)]).content.strip()
        jobs.append({"job_desc": job_desc, "accurate_websites": [w['Website Links'] for w in selected]})
    return jobs

def write_model_analysis_csv(results, filename="Model-Analysis.csv"):
    headers = [
        "Job",
        "Category",
        "Accurate Websites",
        "Upwork Case Study",
        "Figma Link",
        "AI Predicted Website 1",
        "AI Predicted Website 2",
        "AI Predicted Website 3",
        "AI Predicted Website 4",
        "Accuracy Score"
    ]
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for row in results:
            writer.writerow(row)

if __name__ == '__main__':
    # Generate synthetic jobs using Langchain
    # jobs = generate_llm_jobs(n=5)
    # print("\n===== LLM-Generated Job QA Evaluation =====\n")
    # for i, job in enumerate(jobs):
    #     print(f"--- Job {i+1} ---")
    #     job_title = job.get('title', '')
    #     job_desc = job.get('description', '')
    #     print(f"Job Title: {job_title}")
    #     evaluate_website_retrieval(job_desc)
    
    # Generate jobs from actual websites
    # jobs = generate_jobs_from_websites(n=5)
    # print("\n===== Website-Driven Job QA Evaluation =====\n")
    # for i, job in enumerate(jobs):
    #     print(f"--- Job {i+1} ---")
    #     print(f"Job: {job['job_desc']}")
    #     print(f"Accurate Website: {job['accurate_website']}")
    #     # Show top 3 predicted websites
    #     portfolio = get_relevant_portfolio(job['job_desc'], top_k=3)
    #     predicted_websites = [w['Website Links'] for w in portfolio]
    #     print(f"Predicted Websites: {predicted_websites}\n")
    
    # Generate jobs from combinations of websites
    jobs = generate_jobs_from_website_combinations(n=50, k=3)
    print("\n===== Website-Combination Job QA Evaluation =====\n")
    results = []
    for i, job in enumerate(jobs):
        print(f"--- Job {i+1} ---")
        print(f"Job: {job['job_desc']}")
        print(f"Accurate Websites: {job['accurate_websites']}")
        # Predicted websites
        portfolio = get_relevant_portfolio(job['job_desc'], top_k=6)
        predicted_websites = [w['Website Links'] for w in portfolio]
        # Use the category of the first accurate website
        category = ''
        if job['accurate_websites']:
            # Find the category from the embeddings
            _, websites = load_embeddings()
            for w in websites:
                if w['Website Links'] == job['accurate_websites'][0]:
                    category = w.get('Category', '')
                    break
        # Upwork Case Study links
        upwork_portfolio = get_relevant_portfolio_upwork(job['job_desc'], top_k=1)
        upwork_link = upwork_portfolio[0]['Website Link'] if upwork_portfolio else ''
        # Figma link
        figma_portfolio = get_relevant_portfolio_figma(job['job_desc'], top_k=3)
        figma_link = figma_portfolio[0]['Figma Link'] if figma_portfolio else ''
        print(f"Predicted Websites: {predicted_websites}")
        print(f"Upwork Case Study: {upwork_link}")
        print(f"Figma Link: {figma_link}\n")
        # Calculate accuracy score: fraction of accurate websites found in predicted websites
        match_count = sum(1 for w in job['accurate_websites'] if w in predicted_websites)
        accuracy_score = match_count / len(job['accurate_websites']) if job['accurate_websites'] else 0.0
        # Prepare row for CSV
        row = [
            job['job_desc'],
            category,
            '; '.join(job['accurate_websites']),
            upwork_link,
            figma_link
        ] + predicted_websites + [''] * (4 - len(predicted_websites)) + [f"{accuracy_score:.2f}"]
        results.append(row)
    write_model_analysis_csv(results)
    print(f"\nResults saved to Model-Analysis.csv\n")
