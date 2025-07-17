import os
import re
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType, Tool
from langchain.schema import HumanMessage
from sentence_transformers import SentenceTransformer
import pickle
import numpy as np
from langchain.prompts import PromptTemplate
from upwork_data import MY_DATA
import json
from datetime import datetime
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from vector_storage import (
    ProposalHistoryEntry,
    save_proposal_history,
    get_all_history_entries,
    get_st_embeddings,
    retrieve_similar_history
)
import uuid


load_dotenv(find_dotenv())

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
EMBEDDINGS_FILE = './embeddings/website_embeddings_2.pkl'
UPWORK_EMBEDDINGS_FILE = './embeddings/upwork_embeddings.pkl'
FIGMA_EMBEDDINGS_FILE = './embeddings/figma_embeddings.pkl'
VECTOR_STORAGE_DIR = './VectorStorage'
VECTOR_DIM = 384  # all-MiniLM-L6-v2 output dim


def load_embeddings():
    with open(EMBEDDINGS_FILE, 'rb') as f:
        data = pickle.load(f)
    return data['embeddings'], data['websites'], data['texts']


def load_upwork_embeddings():
    with open(UPWORK_EMBEDDINGS_FILE, 'rb') as f:
        data = pickle.load(f)
    return data['embeddings'], data['websites']


def load_figma_embeddings():
    with open(FIGMA_EMBEDDINGS_FILE, 'rb') as f:
        data = pickle.load(f)
    return data['embeddings'], data['figmas']


def get_relevant_portfolio(job_desc, top_k=3, similarity_threshold=0.5):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings, websites, _ = load_embeddings()

    job_emb = model.encode([job_desc])[0]
    similarities = np.dot(embeddings, job_emb) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(job_emb)
    )

    # Get top_k highest scores
    top_indices = similarities.argsort()[-top_k:][::-1]

    results = []
    for idx in top_indices:
        sim_score = similarities[idx]
        if sim_score >= similarity_threshold:
            w = websites[idx]
            results.append({
                'Website URL': w['Website URL'],
                'Category': w.get('Category', ''),
                'Sub Category': w.get('sub category', ''),
                'Niche': w.get('Niche', ''),
                'Priority': w.get('Priority', ''),
                'Description': w.get('Website Description', ''),
                'Similarity Score': round(float(sim_score), 4)
            })

    return results



def get_ranked_portfolio(job_desc):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings, websites, texts = load_embeddings()

    job_emb = model.encode([job_desc])[0]

    similarities = np.dot(embeddings, job_emb) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(job_emb)
    )

    # Step 1: Best contextual match by subcategory or description
    ranked = sorted(
        [(i, s) for i, s in enumerate(similarities) if s >= 0.4],
        key=lambda x: x[1],
        reverse=True
    )

    link1 = None
    for idx, sim in ranked:
        w = websites[idx]
        subcat = w.get('sub category', '').lower()
        desc = w.get('Website Description', '').lower()
        if any(word in job_desc.lower() for word in subcat.split()) or any(word in desc for word in job_desc.lower().split()):
            link1 = (w, sim, 'AI-Priority')
            break

    if not link1 and ranked:
        idx = ranked[0][0]
        sim = ranked[0][1]
        link1 = (websites[idx], sim, 'AI-Priority')

    main_category = link1[0]['Category'] if link1 else None

    def extract_priority(p):
        try:
            return int(p.strip().split('#')[-1])
        except:
            return 99

    same_cat_entries = [
        (w, extract_priority(w.get('Priority', ''))) for w in websites
        if w.get('Category') == main_category
    ]

    same_cat_sorted = sorted(same_cat_entries, key=lambda x: x[1])
    link2 = same_cat_sorted[0][0] if len(same_cat_sorted) > 0 else None
    link3 = same_cat_sorted[1][0] if len(same_cat_sorted) > 1 else None

    def format_result(w, priority_label=None):
        actual_priority = priority_label if priority_label else w.get('Priority', '').strip()
        return (
            f"{w['Website URL']} "
            f"({w.get('Category', '').strip()} – {w.get('sub category', '').strip()} – "
            f"{w.get('Niche', '').strip()} – {actual_priority})\n"
            f"→ {w.get('Website Description', '').strip()}"
        )

    result = []
    if link1:
        result.append(format_result(link1[0], 'AI-Priority'))
    if link2 and link2 != link1[0]:
        result.append(format_result(link2))
    if link3 and link3 != link2 and link3 != link1[0]:
        result.append(format_result(link3))

    return result



def get_relevant_portfolio_upwork(job_desc, top_k=2):
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


def get_relevant_portfolio_figma(job_desc, top_k=2):
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
    portfolio = get_ranked_portfolio(job_desc)
    return '\n\n'.join([f"- {item}" for item in portfolio])


def upwork_portfolio_tool_func(job_desc: str):
    portfolio = get_relevant_portfolio_upwork(job_desc)
    return '\n'.join([f"- {w['Website Link']} ({w['Category']})" for w in portfolio])


def figma_portfolio_tool_func(job_desc: str):
    portfolio = get_relevant_portfolio_figma(job_desc)
    return '\n'.join([f"- {w['Figma Link']} ({w['Category']})" for w in portfolio])


def history_retriever_tool_func(job_desc: str):
    # Retrieve similar history from Pinecone
    entries = retrieve_similar_history(job_desc, top_k=5)
    if not entries:
        return "No relevant history found."
    result = "--- Relevant Past Proposals & Feedback ---\n"
    for i, entry in enumerate(entries):
        result += f"[{i+1}] Date: {entry.date_time}\nJob: {entry.job_text[:100]}...\nProposal: {entry.proposal[:200]}...\nComments: {entry.comments}\nRating: {entry.response_review}\n\n"
    return result



portfolio_tool = Tool(
    name="PortfolioRetriever",
    func=portfolio_tool_func,
    description="Given a job description, returns 3 relevant portfolio links from the matching niche."
)
upwork_portfolio_tool = Tool(
    name="UpworkPortfolioRetriever",
    func=upwork_portfolio_tool_func,
    description="Given a job description, returns 1 relevant Upwork Case Study link from the matching niche."
)
figma_portfolio_tool = Tool(
    name="FigmaPortfolioRetriever",
    func=figma_portfolio_tool_func,
    description="Given a job description, returns 1 relevant Figma Portfolio link from the matching niche."
)

history_retriever_tool = Tool(
    name="HistoryRetriever",
    func=history_retriever_tool_func,
    description="Given a job description, returns relevant past proposals and feedback from Pinecone history."
)


def get_proposal_agent():
    llm = ChatOpenAI(model="gpt-4.1-mini", api_key=OPENAI_API_KEY)
    agent = initialize_agent(
        tools=[portfolio_tool, upwork_portfolio_tool, figma_portfolio_tool, history_retriever_tool],
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True,
        verbose=True,
        max_iterations=40,
        max_execution_time=180,
        early_stopping_method="generate"
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
    1. Introduction: Write a 2-short lines introduction using the following rule:
        Use this base:  ({MY_DATA['profile']['profile_views']}+ earned on Upwork | {MY_DATA['profile']['num_of_jobs']}+ projects | {MY_DATA['profile']['years_of_experience']}+ years of experience)
        Always mention Upwork earnings
        Align emotionally with client’s industry or goal
        Speak like a confident business owner (not a junior freelancer)
        No technical terms or platform names in the intro
    e.g: "I’ve built dozens of websites for fitness brands, coaches, and consultants focused on growth. With over $400K earned on Upwork, I bring strategy and execution that moves results."

    2. Understanding the Project: In the second paragraph, clearly reference the client's goals and pain points (without rewording the job post). Explain why this project excites you, and avoid listing tech terms or basic features.
    3. Use the Upwork Portfolio Retriever tool to include 1 relevant Upwork Case Study link ONLY from the matching niche. Only include the raw link, not markdown or titles. Format them as bullets:
    • [raw link]
    
    4. If the job description does not include any specific niche then use the Best_Portfolio. Otherwise, use the Portfolio Retriever tool to include 2-3 relevant portfolio links ONLY from the matching niche. Only include the raw link, not markdown or titles. Format them as bullets:
    • [raw link]
    [summary of raw link]
    • [raw link]
    [summary of raw link]
    • [raw link]
    [summary of raw link]
    ...
    5. If the job description includes "Figma" or "figma", use the Figma Portfolio Retriever tool. Mention, "This is the website we have designed and the prototype link is [prototype link]. This is what we have developed [website link]." Ensure the project name matches. Only include the raw link, not markdown or titles.
    6. If the job description does not include "Figma" or "figma", use the Figma Portfolio Retriever tool to include the Figma link of the relevant category design prototype. Only include the raw link, not markdown or titles.
    7. Also write the short summary of the Figma link in the same format as above.
    8. Follow the Creasions writing style (calm, confident, helpful): Avoid sounding like a resume, avoid overselling or listing buzzwords, and show excitement for the client's niche. Never list technical skills or tool names as qualifications.
    9. Give timeline or delivery estimate if possible. If not mentioned, suggest a confident estimate (e.g., 2–4 weeks).
    10. Always close with: "Let's schedule a call to discuss this project further and explore how I can contribute to your success."
    11. Avoid discussing budget or pricing details in the proposal text, regardless of their presence in the job description.
    12. Do not add any extra line before or after the proposal text.

    Note: Ensure the final proposal is humanized. Aim for a personal, friendly, and approachable style, avoiding robotic or generic language, as well as emojis or special characters like "–" which are often used by AI.

'''


Best_Portfolio = '''
Designs:
• https://www.creasions.com/portfolio/website-design-1
• https://www.creasions.com/portfolio/website-design-2
• https://www.creasions.com/portfolio/website-design-3


Development:
• https://www.creasions.com/portfolio/website-development-1
• https://www.creasions.com/portfolio/website-development-2
• https://www.creasions.com/portfolio/website-development-3

UI/UX or Figma Prototypes:
• https://www.figma.com/proto/1234567890abcdef/Project-Name?node-id=1%3A2
• https://www.figma.com/proto/abcdef1234567890/Another-Project?node-id=3%3A4
• https://www.figma.com/proto/0987654321fedcba/Third

Upwork Case Studies:
• https://www.upwork.com/case-study/1234567890abcdef
• https://www.upwork.com/case-study/abcdef1234567890


'''

def humanize_proposal(proposal):
    llm = ChatOpenAI(model="gpt-4.1-mini", api_key=OPENAI_API_KEY)
    humanize_prompt = f"""
    Rewrite the following proposal to make it sound like it was written by a friendly and empathetic human freelancer. 
    - Keep the same headings, bullet points, and formatting.
    - Use a natural and conversational tone.
    - Add warmth and personality to the proposal where appropriate.
    - Do not add or remove sections; keep the structure and length similar.
    - Examples of robotic language: 'I am interested in your job. I have experience.'
    - Examples of humanized language: 'I'm excited about your project and have helped clients with similar needs before.'
    - Always close with: "Let's schedule a call to discuss this project further and explore how I can contribute to your success."
    
    Important:
    - Avoid using robotic, generic, or overly formal language.
    - Do not use emojis or special characters like —, –, etc.


    Original Proposal:
    {proposal}

    Humanized Proposal:
    """
    humanized_proposal = llm([HumanMessage(content=humanize_prompt)]).content.strip()

    print("\n\n" + "="*50 + " HUMANIZED PROPOSAL " + "="*50)
    print(humanized_proposal)
    print("\n" + "="*50 + " END OF HUMANIZED PROPOSAL " + "="*40 + "\n\n")
    return humanized_proposal


# Function to retrieve and display relevant history and feedback

def show_relevant_history_and_feedback(job_text, top_k=5):
    """
    Retrieve and display relevant past proposals and feedback for the given job description.
    """
    entries = retrieve_similar_history(job_text, top_k=top_k)
    if not entries:
        print("No relevant history found.")
        return []
    print("\n--- Relevant Past Proposals & Feedback ---\n")
    for i, entry in enumerate(entries):
        print(f"[{i+1}] Date: {entry.date_time}")
        print(f"Job: {entry.job_text[:100]}...")
        print(f"Proposal: {entry.proposal[:200]}...")
        print(f"Comments: {entry.comments}")
        print(f"Rating: {entry.response_review}")
        print()
    return entries


# Main proposal agent
def write_proposal(job_text):
    history_entries = show_relevant_history_and_feedback(job_text, top_k=3)
    history_context = "\n--- Relevant Past Proposals & Feedback ---\n"
    for i, entry in enumerate(history_entries):
        history_context += f"[{i+1}] Date: {entry.date_time}\nJob: {entry.job_text[:100]}...\nProposal: {entry.proposal[:200]}...\nComments: {entry.comments}\nRating: {entry.response_review}\n\n"
    if not history_entries:
        history_context += "No relevant history found.\n"


    data = extract_client_job_data(job_text)
    tone_num, tone_name, tone_score = select_tone_vectorized_factors(data)
    print("\n\n", "="*50, "EXTRACTED DATA AND SELECTED TONE", "="*50)
    print("Client Total Spend:", data.get('spend', 'N/A'))
    print("Average Hourly Rate:", data.get('avg_hourly', 'N/A'))
    print("Budget:", data.get('budget', 'N/A'), "(", data.get('budget_type', 'N/A'), ")")
    print("Niche-specific request:", data.get('niche', 'None found'))
    print(f"Selected Tone: Tone {tone_num} - {tone_name}")
    print("\n" + "="*50 + " END OF EXTRACTED DATA AND SELECTED TONE " + "="*50 + "\n\n\n")


    agent = get_proposal_agent()
    llm = ChatOpenAI(model="gpt-4.1-nano", api_key=OPENAI_API_KEY)

    
    agent_instructions = """
        You must always use the following format for every reasoning step:
        Thought: [your reasoning]
        Action: [the tool to use, e.g., HistoryRetriever[<input>], PortfolioRetriever[<input>], UpworkPortfolioRetriever[<input>], FigmaPortfolioRetriever[<input>]]
        Observation: [result of the action]
        ... (repeat as needed)
        When you have gathered all necessary information and completed all tool actions, only then provide:
        Final Answer: [your final proposal]
        Do NOT skip the Action step after any Thought. Do NOT jump directly to Final Answer or justification after a Thought. Every Thought must be followed by an Action and Observation.
        """

    intro_prompt = f"""
    Write a short, single-line personalized intro for an Upwork proposal. The intro should highlight why Muhammad is suitable for the job described below, referencing relevant experience, skills, and achievements from MY_DATA. Be specific to the job context. The intro should be 1-2 sentences long.

    MY_DATA:
    {MY_DATA}

    Job Description:
    {job_text}

    Intro:
    """
    llm = ChatOpenAI(model="gpt-4.1-nano", api_key=OPENAI_API_KEY)
    my_intro = llm([HumanMessage(content=intro_prompt)]).content.strip()
    thoughts = f"""
        {my_intro}
        Job: {job_text}
        Client's total spend: {data.get('spend', 'N/A')}
        Average hourly rate: {data.get('avg_hourly', 'N/A')}
        Niche-specific request: {data.get('niche', 'None found')}
        Selected Tone: Tone {tone_num} - {tone_name}
        Justification for tone: [The agent chose this tone because of the above data.]
        Instructions for Proposal Writing: {PROPOSAL_RULES}
        Best Portfolio Links (for reference):\n{Best_Portfolio}\n
        PROPOSAL BODY STRUCTURE:
        ```
        Start with your personalized 2-line intro
        Brief alignment with client’s vision
        2–3 portfolio links with strong relevance
        Brief description of your approach (outcome-focused, not technical)
        Timeline or delivery estimate (suggest if not mentioned)
        Call to Action:
        “Let’s schedule a call to discuss this project further and explore how I can contribute to your success.”
        ```
        MY_DATA (for reference):\n{MY_DATA}\n
        ---
        Agent Instructions (ReAct format):\n{agent_instructions}\n
        {history_context}
    """


    prompt_template = PromptTemplate(
        input_variables=["thoughts"],
        template="""
You are Muhammad, an expert Upwork freelancer and proposal writer agent. Your job is to generate personalized Upwork proposals that sound human, confident, and tailored — never robotic.
Below are your thoughts and observations for the provided job data:

{thoughts}

Strictly follow this reasoning format for every step:
Thought: [your reasoning]
Action: [the tool to use, e.g., HistoryRetriever[<input>], PortfolioRetriever[<input>], UpworkPortfolioRetriever[<input>], FigmaPortfolioRetriever[<input>]]
Observation: [result of the action]
... (repeat as needed)
When you have gathered all necessary information and completed all tool actions, only then provide:
Final Answer: [your full proposal text]
Do NOT skip the Action step after any Thought. Do NOT jump directly to Final Answer or justification after a Thought. Every Thought must be followed by an Action and Observation.

After the Final Answer, provide additional context for the agent's thought process as the "Justification of Proposal". This should be a short description (less than 300 characters) of:
- A brief justification for the paragraphs you have written.
- Why selected links are relevant and suitable for the job, including a justification for each link.
- Why some links were not selected (if not selected) and provide the link of them in justification too, including a justification for each link.
- Any other relevant context or notes for the agent's thought process.

Return your output in the following pattern: 
Final Answer: [your full proposal text]
---
Justification of tone: [your justification for the selected tone]
---
Justification of Proposal: [your justification for the proposal, including why you wrote each paragraph, why you selected certain links, and any other relevant context]
"""
    )

    user_prompt = prompt_template.format(thoughts=thoughts)
    print("Extracted Data:", data)
    print(f"Selected Tone: Tone {tone_num} - {tone_name} (scores: {tone_score})")
    proposal = agent.run(user_prompt)

    # Humanize the proposal
    final_output = humanize_proposal(proposal)
    # Checklist evaluation loop
    while True:
        checklist_results = checklist_tool(job_text, final_output)
        failed_items = [item for item, passed in checklist_results.items() if not passed]
        if not failed_items:
            break
        corrections = f"Please correct the following issues: {failed_items}"
        final_output = proposal_corrector(final_output, corrections)
    proposal_id = str(uuid.uuid4())
    entry = ProposalHistoryEntry(
        date_time=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        job_text=job_text,
        proposal=final_output,
        comments="",
        response_review=None,
        proposal_id=proposal_id
    )

    final_Proposal = replace_with_humanized(proposal,final_output)
    
    print("\n\n", "="*50, "FINAL PROPOSAL", "="*50)
    print(final_Proposal)
    print("\n" + "="*50 + " END OF FINAL PROPOSAL " + "="*40 + "\n\n\n")
    save_proposal_history(entry)
    return final_Proposal, proposal_id



def replace_with_humanized(proposal, humanized_proposal):
    """
    Replaces the body of the proposal with the humanized version, while preserving the justification headings if present.
    If the expected format is missing, returns the humanized proposal as-is.
    """
    parts = proposal.split('---')
    # Remove empty strings and strip whitespace
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) < 3 or len(parts) < 2:
        return humanized_proposal
    final_answer = parts[0] if parts[0].lower().startswith('final answer:') else ''
    tone_justification = ''
    proposal_justification = ''
    for p in parts:
        if p.lower().startswith('justification of tone:'):
            tone_justification = p
        elif p.lower().startswith('justification of proposal:'):
            proposal_justification = p
    result = ''
    if tone_justification:
        result += f"---\n{tone_justification}\n"
    if final_answer:
        result += f"---\n\n{humanized_proposal}\n"
    else:
        result += f"\n{humanized_proposal}\n"
    if proposal_justification:
        result += f"---\n{proposal_justification}\n---"
    return result.strip()


def checklist_tool(job_text, proposal):
    """
    Returns a dict of checklist items and their pass/fail status for the proposal.
    """
    checklist = [
        "Tone is accurate based on budget and niche",
        "$400K+ earnings always mentioned in intro",
        "Only validated portfolio links used",
        "Figma logic followed",
        "Timeline is stated",
        "Strong CTA included",
        "Make sure the proposal is humanized",
        "Avoid using robotic, generic, or overly formal language."
        "Do not use emojis or special characters like —, –, etc."

    ]
    llm = ChatOpenAI(model="gpt-4.1-nano", api_key=OPENAI_API_KEY)
    eval_prompt = f"""
You are a Proposal Checklist Agent. For each checklist item below, return True if the proposal meets it, otherwise False. Return a JSON object with checklist items as keys and True/False as values.
Checklist:
{json.dumps(checklist)}

PROPOSAL RULES:
{PROPOSAL_RULES}

Job Description:
{job_text}
Proposal:
{proposal}
Return format:
{{"Tone is accurate based on budget and niche": true/false, ...}}
"""
    response = llm([HumanMessage(content=eval_prompt)]).content.strip()
    print("\n\n", "="*50, "CHECKLIST RESPONSE", "="*50)
    print(response)
    print("\n" + "="*50 + " END OF CHECKLIST RESPONSE " + "="*50 + "\n\n\n")

    try:
        result = json.loads(response)
    except Exception:
        result = {item: False for item in checklist}
    return result

def proposal_corrector(proposal, corrections, job_text=None):
    """
    Resolves only the specific failed checklist items in the proposal.
    If Upwork, Portfolio, Websites, or Figma link is missing, fetch and insert them using relevant functions.
    For other items, prompt LLM to rewrite only the failed section(s).
    """
    updated_proposal = proposal
    if job_text and corrections:
        # corrections is a list of failed items
        for item in corrections:
            if "Portfolio" in item or "Websites" in item:
                # Insert portfolio links
                portfolio_links = portfolio_tool_func(job_text)
                # Replace or append portfolio section
                if "•" in updated_proposal:
                    # Replace existing portfolio links
                    updated_proposal = re.sub(r"(•.*?)(\n\n|$)", f"{portfolio_links}\n\n", updated_proposal, flags=re.DOTALL)
                else:
                    updated_proposal += f"\n{portfolio_links}\n"
            elif "Figma" in item:
                figma_links = figma_portfolio_tool_func(job_text)
                if "figma.com" in updated_proposal:
                    updated_proposal = re.sub(r"(https://www.figma.com/.*?)(\n|$)", f"{figma_links}\n", updated_proposal)
                else:
                    updated_proposal += f"\n{figma_links}\n"
            elif "Upwork" in item:
                upwork_links = upwork_portfolio_tool_func(job_text)
                if "upwork.com" in updated_proposal:
                    updated_proposal = re.sub(r"(https://www.upwork.com/.*?)(\n|$)", f"{upwork_links}\n", updated_proposal)
                else:
                    updated_proposal += f"\n{upwork_links}\n"
            else:
                # For other items, ask LLM to rewrite only the failed section
                llm = ChatOpenAI(model="gpt-4.1-mini", api_key=OPENAI_API_KEY)
                correct_prompt = f"""
You are a Proposal Correction Agent. Here is a proposal. Only rewrite the section that addresses the following failed checklist item, keeping the rest unchanged.
Proposal:
{updated_proposal}
Failed Item:
{item}
Return only the corrected proposal text.
"""
                updated_proposal = llm([HumanMessage(content=correct_prompt)]).content.strip()
    return updated_proposal



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
    job_title = "Coaching Website for Online Wellness Program"
    job_desc = """
            I’m a wellness coach launching an online program. Need a landing site with booking calendar, testimonials, video content, blog, and integration with Teachable or Kajabi.
        """
    proposal, proposal_id = write_proposal(job_desc)
    print(f"Proposal ID: {proposal_id}")
    print(f"Generated Proposal:\n{proposal}")
