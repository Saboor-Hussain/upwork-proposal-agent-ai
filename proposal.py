import os
import re
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType, Tool
from langchain.schema import HumanMessage
from sentence_transformers import SentenceTransformer
import pickle
import numpy as np
from langchain.prompts import PromptTemplate

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
EMBEDDINGS_FILE = './embeddings/website_embeddings_1.pkl'
UPWORK_EMBEDDINGS_FILE = './embeddings/upwork_embeddings.pkl'
FIGMA_EMBEDDINGS_FILE = './embeddings/figma_embeddings.pkl'


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


def get_relevant_portfolio(job_desc, top_k=3):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings, websites = load_embeddings()
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


def get_relevant_portfolio_upwork(job_desc, top_k=1):
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


def get_relevant_portfolio_figma(job_desc, top_k=1):
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
    portfolio = get_relevant_portfolio(job_desc)
    return '\n'.join([f"- {w['Website Links']} ({w['Category']})" for w in portfolio])


def upwork_portfolio_tool_func(job_desc: str):
    portfolio = get_relevant_portfolio_upwork(job_desc)
    return '\n'.join([f"- {w['Website Link']} ({w['Category']})" for w in portfolio])


def figma_portfolio_tool_func(job_desc: str):
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
    description="Given a job description, returns 1 relevant Upwork Case Study link from the matching niche."
)
figma_portfolio_tool = Tool(
    name="Figma Portfolio Retriever",
    func=figma_portfolio_tool_func,
    description="Given a job description, returns 1 relevant Figma Portfolio link from the matching niche."
)


def get_proposal_agent():
    llm = ChatOpenAI(model="gpt-4.1-mini", api_key=OPENAI_API_KEY)
    agent = initialize_agent(
        tools=[portfolio_tool, upwork_portfolio_tool, figma_portfolio_tool],
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True,
        verbose=False
    )
    return agent


# Helper to extract client/job data from raw job text
def extract_client_job_data(job_text):
    """
    Use LLM to extract spend, avg hourly, budget, budget type, and niche from raw job text.
    """
    llm = ChatOpenAI(model="gpt-4.1-nano", api_key=OPENAI_API_KEY)
    extraction_prompt = f"""
Extract the following fields from the job post below. If a field is not present, return null for it.
Return your answer as a JSON object with these keys: spend, avg_hourly, budget, budget_type, niche.

Job Post:
{job_text}

Fields to extract:
- spend: Client's total spend (e.g., $9.1k, $2,800, $9100)
- avg_hourly: Average hourly rate paid by the client (e.g., $27.00/hr)
- budget: Job budget (number only, e.g., 2800 or 27)
- budget_type: 'fixed' or 'hourly'
- niche: Any niche-specific request (e.g., 'gym websites', 'legal branding', or null)

JSON:
"""
    response = llm([HumanMessage(content=extraction_prompt)]).content
    import json
    try:
        data = json.loads(response)
    except Exception:
        # Fallback: try to extract JSON from the response
        import re
        match = re.search(r'\{.*\}', response, re.DOTALL)
        if match:
            data = json.loads(match.group(0))
        else:
            data = {'spend': None, 'avg_hourly': None, 'budget': None, 'budget_type': None, 'niche': None}
    return data



def parse_money(val):
    if not val:
        return 0.0
    s = str(val).replace('$','').replace(',','').strip()
    if s.lower().endswith('k'):
        try:
            return float(s[:-1]) * 1000
        except Exception:
            return 0.0
    try:
        return float(s)
    except Exception:
        return 0.0


def parse_float(val):
    try:
        return float(val)
    except Exception:
        return 0.0


def select_tone_vectorized_factors(data):
    """
    Use embeddings to match extracted factors to tone archetypes and select the tone with the most matching factors.
    """
    # Each tone archetype has a set of factors
    TONE_FACTORS = [
        {
            'name': 'Normal Proposal',
            'factors': [
                lambda d: d.get('spend') and parse_money(d['spend']) < 2000,
                lambda d: d.get('avg_hourly') and parse_float(d['avg_hourly']) < 20,
                lambda d: d.get('budget_type') == 'fixed' and parse_float(d.get('budget', 0)) < 500,
                lambda d: d.get('budget_type') == 'hourly' and parse_float(d.get('budget', 0)) < 20,
            ]
        },
        {
            'name': 'Confident Expert Proposal',
            'factors': [
                lambda d: d.get('avg_hourly') and parse_float(d['avg_hourly']) >= 30,
                lambda d: d.get('budget_type') == 'hourly' and parse_float(d.get('budget', 0)) >= 25,
                lambda d: d.get('budget_type') == 'fixed' and parse_float(d.get('budget', 0)) >= 1000,
                lambda d: d.get('spend') and parse_money(d['spend']) > 2000,
            ]
        },
        {
            'name': 'Specialized Proposal',
            'factors': [
                lambda d: d.get('niche') is not None and d.get('niche') != 'None found',
            ]
        }
    ]
    # Count matches for each tone
    scores = []
    for tone in TONE_FACTORS:
        score = sum(1 for f in tone['factors'] if f(data))
        scores.append(score)
    best_idx = int(np.argmax(scores))
    return best_idx+1, TONE_FACTORS[best_idx]['name'], scores



# Tone selection logic
def select_tone(data):
    """
    Decide tone based on extracted factors, with clear boundary logic.
    """
    spend = float(str(data.get('spend', '0')).replace('$','').replace('k','000')) if data.get('spend') else 0
    avg_hourly = data['avg_hourly'] or 0
    budget = data['budget'] or 0
    budget_type = data['budget_type']
    niche = data['niche']
    # Tone 3: Specialized
    if niche:
        return 3, f"Specialized Proposal: Detected niche-specific request ('{niche}'). Ultra-tailored response."
    # Tone 2: Confident Expert
    if (avg_hourly >= 30) or (budget_type == 'hourly' and budget >= 25) or (budget_type == 'fixed' and budget >= 1000):
        return 2, "Confident Expert Proposal: High avg hourly or budget. Position as expert partner."
    # Tone 1: Normal
    if (spend < 2000) or (avg_hourly < 20) or (budget_type == 'fixed' and budget < 500) or (budget_type == 'hourly' and budget < 20):
        return 1, "Normal Proposal: Lower spend or budget. Clear, simple, informative tone."
    # Default
    return 1, "Normal Proposal: Default."


# Proposal writing rules
PROPOSAL_RULES = '''
Write a personalized Upwork proposal for the following job. Follow these rules:
1. Start with a personalized greeting.
2. Mention the job/niche context in the first line.
3. Use the Upwork Portfolio Retriever tool to include 1–2 relevant Upwork Case Study links ONLY from the matching niche. Only include the raw link, not markdown or titles.
4. Use the Portfolio Retriever tool to include 2–3 relevant portfolio links ONLY from the matching niche. Only include the raw link, not markdown or titles. Format them as bullets:
  • [raw link]
  • [raw link]
  • [raw link]
  ...
5. If the job description includes "Figma" or "figma", use the Figma Portfolio Retriever tool. Mention, "This is the website we have designed and the prototype link is [prototype link]. This is what we have developed [website link]." Ensure the project name matches. Only include the raw link, not markdown or titles.
6. If the job description does not include "Figma" or "figma", use the Figma Portfolio Retriever tool to include the Figma link of the relevant category design prototype. Only include the raw link, not markdown or titles.
7. Follow the Creasions writing style (calm, confident, helpful).
8. Mention tech/tools if listed in the job.
9. Give timeline or delivery estimate if possible.
10. End with a CTA (e.g., let’s connect, would love to chat).
11. Do NOT mention budget or pricing in the proposal text, even if present in the job description.
'''


# Main proposal agent
def write_proposal(job_text):
    data = extract_client_job_data(job_text)
    tone_num, tone_name, tone_score = select_tone_vectorized_factors(data)
    # Extracted data and selected tone
    print("\n\n", "="*50, "EXTRACTED DATA AND SELECTED TONE", "="*50)
    print("Client Total Spend:", data.get('spend', 'N/A'))
    print("Average Hourly Rate:", data.get('avg_hourly', 'N/A'))
    print("Budget:", data.get('budget', 'N/A'), "(", data.get('budget_type', 'N/A'), ")")
    print("Niche-specific request:", data.get('niche', 'None found'))
    print(f"Selected Tone: Tone {tone_num} - {tone_name}")
    print("\n" + "="*50 + " END OF EXTRACTED DATA AND SELECTED TONE " + "="*50 + "\n\n\n")


    agent = get_proposal_agent()
    thoughts = f"""
    Job: {job_text}
    Client's total spend: {data.get('spend', 'N/A')}
    Average hourly rate: {data.get('avg_hourly', 'N/A')}
    Niche-specific request: {data.get('niche', 'None found')}
    Selected Tone: Tone {tone_num} - {tone_name}=
    Instructions for Proposal Writing: {PROPOSAL_RULES}
    """
    prompt_template = PromptTemplate(
        input_variables=["thoughts"],
        template="""
You are Muhammad, an expert Upwork freelancer and proposal writer agent.
Below are your thoughts and observations for the provided job data:

{thoughts}

Write a highly relevant, tailored Upwork proposal for this job, using the selected tone and all available context. 
"""
    )
    user_prompt = prompt_template.format(thoughts=thoughts)
    print("Extracted Data:", data)
    print(f"Selected Tone: Tone {tone_num} - {tone_name} (scores: {tone_score})")
    proposal = agent.run(user_prompt)
    print("\n--- PROPOSAL ---\n")
    print(proposal)
    return proposal


def get_best_proposal_selector_agent():
    llm = ChatOpenAI(model="gpt-4.1-mini", api_key=OPENAI_API_KEY)

    # Dummy tool to satisfy agent initialization
    dummy_tool = Tool(
        name="noop",
        func=lambda x: "This is a dummy tool. Not used.",
        description="Does nothing. Placeholder tool."
    )

    selector_prompt = """
You are a Senior Proposal Evaluation Agent.

Your job is to carefully evaluate 3 proposals written for an Upwork job and select the best one.

You will receive:
- A job description
- Proposal 1
- Proposal 2
- Proposal 3

Instructions:
- Choose the proposal that is most tailored to the job.
- Check relevance of portfolios and case studies.
- Evaluate clarity, tone, structure, and match to tools/tech mentioned.
- Follow Upwork best practices.
- Justify your choice clearly.
- Proposal must have Upwork, Figma, and portfolio links relevant to the job.

Return your answer in this format:

Best Proposal: Proposal X  
Reason: [short paragraph explaining why this proposal is better than the other two]  
Content:  
[full proposal text here]
"""

    agent = initialize_agent(
        tools=[dummy_tool],
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True,
        verbose=False
    )

    return agent, selector_prompt


def select_best_proposal(job_text, proposal_1, proposal_2, proposal_3):
    agent, selector_prompt = get_best_proposal_selector_agent()

    thoughts = f"""
Job Description:
{job_text}

--- Proposal 1 ---
{proposal_1}

--- Proposal 2 ---
{proposal_2}

--- Proposal 3 ---
{proposal_3}
"""

    prompt = f"{selector_prompt}\n{thoughts}"

    result = agent.run(prompt)

    print("\n--- BEST PROPOSAL EVALUATION ---\n")
    print(result)
    return result


if __name__ == "__main__":
    # Example usage
    job = """
WordPress Site for Local Roofing Company
Posted: 1 week ago — Location: United States
Need a small business website for a roofing company to showcase services, projects, and collect leads. Must include quote form, map, and FAQs.
Deliverables:
Homepage, Services, Gallery, Quote Request Form
Testimonials, FAQs, Google Map
Tools: WordPress, Elementor, WPForms
Budget: $1,600 fixed
 Duration: 2–3 weeks
 Expertise: Entry-Intermediate
About the Client:
 Payment verified
 Rating: 4.7 of 5 reviews
 USA
 6 jobs posted, 4 hires
 $2.1K spent
 Avg hourly: $21/hr
"""

    print("\n--- Generating 3 proposals ---\n")
    proposal1 = write_proposal(job)
    print(proposal1)
    # proposal2 = write_proposal(job)
    # proposal3 = write_proposal(job)
    # print("\n--- Selecting the best proposal ---\n")
    # select_best_proposal(job, proposal1, proposal2, proposal3)
