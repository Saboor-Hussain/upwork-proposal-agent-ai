import os
import pickle
import numpy as np
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType, Tool
from langchain.callbacks import get_openai_callback
from langchain.schema import HumanMessage
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv


load_dotenv()
EMBEDDINGS_FILE = './embeddings/website_embeddings_1.pkl'
UPWORK_EMBEDDINGS_FILE = './embeddings/upwork_embeddings.pkl'
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

def load_embeddings():
    with open(EMBEDDINGS_FILE, 'rb') as f:
        data = pickle.load(f)
    return data['embeddings'], data['websites']

def load_upwork_embeddings():
    with open(UPWORK_EMBEDDINGS_FILE, 'rb') as f:
        data = pickle.load(f)
    return data['embeddings'], data['websites']

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



def portfolio_tool_func(job_desc: str):
    """Tool to retrieve relevant portfolio links for a job description."""
    portfolio = get_relevant_portfolio(job_desc)
    return '\n'.join([f"- {w['Website Links']} ({w['Category']})" for w in portfolio])


def upwork_portfolio_tool_func(job_desc: str):
    """Tool to retrieve relevant Upwork portfolio links for a job description."""
    portfolio = get_relevant_portfolio_upwork(job_desc)
    return '\n'.join([f"- {w['Website Link']} ({w['Category']})" for w in portfolio])


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
        5. Follow the Creasions writing style (calm, confident, helpful).
        6. Mention tech/tools if listed in the job: {', '.join(techs)}
        7. Give timeline or delivery estimate if possible: {delivery_estimate}
        8. End with a CTA (e.g., let’s connect, would love to chat).

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

if __name__ == '__main__':
    main()
