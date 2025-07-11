import pandas as pd
import time
import re
from proposal import write_proposal, select_best_proposal
from langchain.callbacks import get_openai_callback
import openpyxl

data_file = './data/jobs_data.csv'
output_file = 'proposal-analysis.xlsx'

def extract_links(text, pattern):
    return re.findall(pattern, text)

def extract_proposal_content(best_result):
    # If dict, try to get 'Content' or 'content' key, else fallback
    proposal = ''
    if isinstance(best_result, dict):
        for k in ['Content', 'content', 'proposal', 'best_proposal']:
            if k in best_result:
                proposal = best_result[k]
                break
        if not proposal and 'Content:' in str(best_result):
            proposal = str(best_result).split('Content:')[1].strip()
        if not proposal:
            proposal = str(best_result)
    elif isinstance(best_result, str):
        if 'Content:' in best_result:
            proposal = best_result.split('Content:')[1].strip()
        else:
            proposal = best_result
    # Remove leading/trailing --- and whitespace
    proposal = proposal.strip()
    if proposal.startswith('---'):
        proposal = proposal.lstrip('-').strip()
    if proposal.endswith('---'):
        proposal = proposal.rstrip('-').strip()
    return proposal

def extract_specific_links(proposal_text):
    # Upwork Portfolio
    upwork_links = extract_links(proposal_text, r'https://www\.upwork\.com/[\w\-/\?=&%\.]+')
    upwork_portfolio = upwork_links[0] if upwork_links else ''
    # Website links
    website_links = extract_links(proposal_text, r'https?://[\w\.-]+(?:/[^\s\)]*)?')
    # Remove upwork and figma links from website_links
    website_links = [l for l in website_links if 'upwork.com' not in l and 'figma.com' not in l]
    website_links = website_links[:3] + [''] * (3 - len(website_links))
    # Figma link
    figma_links = extract_links(proposal_text, r'https://www\.figma\.com/[\w\-/\?=&%\.]+')
    figma_link = figma_links[0] if figma_links else ''
    return upwork_portfolio, website_links, figma_link

df = pd.read_csv(data_file, header=None)
jobs = df[0].dropna().tolist()
total_rows = len(jobs)

rows = []
columns = ["Job", "Proposal", "Upwork Portfolio", "Websites link 1", "Websites link 2", "Websites link 3", "Figma Link", "Total Tokens"]
for idx, job_text in enumerate(jobs, 1):
    print(f"Processing {idx} out of {total_rows}.")
    proposals = []
    total_tokens = 0
    with get_openai_callback() as cb:
        for _ in range(3):
            proposal = write_proposal(job_text)
            proposals.append(proposal)
        total_tokens += cb.total_tokens
    best_result = select_best_proposal(job_text, proposals[0], proposals[1], proposals[2])
    proposal_text = extract_proposal_content(best_result)
    upwork_portfolio, website_links, figma_link = extract_specific_links(proposal_text)
    row_out = [
        job_text,
        proposal_text,
        upwork_portfolio,
        website_links[0],
        website_links[1],
        website_links[2],
        figma_link,
        total_tokens
    ]
    rows.append(row_out)
    # Write/update Excel after each row
    pd.DataFrame(rows, columns=columns).to_excel(output_file, index=False)
    print(f"Row written for job: {job_text[:40]}...")
    time.sleep(10)

print(f"Results saved to {output_file}")
