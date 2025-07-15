import os
import json
from datetime import datetime
from sentence_transformers import SentenceTransformer
import uuid
from pinecone import Pinecone, ServerlessSpec

VECTOR_DIM = 384  # all-MiniLM-L6-v2 output dim
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "upwork-proposals"
PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"

class ProposalHistoryEntry:
    def __init__(self, date_time, job_text, proposal, comments, response_review, proposal_id=None):
        self.date_time = date_time
        self.job_text = job_text
        self.proposal = proposal
        self.comments = comments if comments is not None else ""
        self.response_review = response_review if response_review is not None else ""
        self.proposal_id = proposal_id or str(uuid.uuid4())
    def to_dict(self):
        return {
            'date_time': self.date_time,
            'job_text': self.job_text,
            'proposal': self.proposal,
            'comments': self.comments if self.comments is not None else "",
            'response_review': self.response_review if self.response_review is not None else "",
            'proposal_id': self.proposal_id
        }
    @staticmethod
    def from_dict(d):
        return ProposalHistoryEntry(
            d['date_time'], d['job_text'], d['proposal'], d.get('comments', ""), d.get('response_review', ""), d.get('proposal_id')
        )

def get_st_embeddings():
    return SentenceTransformer('all-MiniLM-L6-v2')

# --- Pinecone setup ---
def pinecone_init():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    # Create index if not exists
    if PINECONE_INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=VECTOR_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION)
        )
    index = pc.Index(PINECONE_INDEX_NAME)
    return index

# --- Save proposal history and vector ---
def save_proposal_history(entry: ProposalHistoryEntry):
    index = pinecone_init()
    model = get_st_embeddings()
    text = entry.job_text + ' ' + entry.proposal + ' ' + (entry.comments or '')
    vec = model.encode([text])[0].tolist()
    meta = entry.to_dict()
    # Upsert to Pinecone
    index.upsert([(entry.proposal_id, vec, meta)])
    track_proposal_id(entry.proposal_id)

# --- Get all history entries ---
def get_all_history_entries():
    ids_file = "proposal_ids.json"
    entries = []
    if os.path.exists(ids_file):
        with open(ids_file, 'r', encoding='utf-8') as f:
            ids = json.load(f)
        index = pinecone_init()
        for pid in ids:
            res = index.fetch([pid])
            for v in res.vectors.values():
                meta = v.metadata
                entries.append(ProposalHistoryEntry.from_dict(meta))
    return entries

# --- Update proposal history by ID ---
def update_proposal_history_by_id(proposal_id, comments, response_review):
    index = pinecone_init()
    res = index.fetch([proposal_id])
    if proposal_id in res.vectors:
        meta = res.vectors[proposal_id].metadata
        meta['comments'] = comments
        meta['response_review'] = response_review
        vec = res.vectors[proposal_id].values
        index.upsert([(proposal_id, vec, meta)])
        return True
    return False

# --- Retrieve similar history ---
def retrieve_similar_history(job_text, top_k=3):
    index = pinecone_init()
    model = get_st_embeddings()
    query_vec = model.encode([job_text])[0].tolist()
    res = index.query(vector=query_vec, top_k=top_k, include_metadata=True)
    results = []
    for match in res['matches']:
        meta = match['metadata']
        results.append(ProposalHistoryEntry.from_dict(meta))
    return results

# --- Track proposal IDs locally for demo ---
def track_proposal_id(proposal_id):
    ids_file = "proposal_ids.json"
    ids = []
    if os.path.exists(ids_file):
        with open(ids_file, 'r', encoding='utf-8') as f:
            ids = json.load(f)
    if proposal_id not in ids:
        ids.append(proposal_id)
        with open(ids_file, 'w', encoding='utf-8') as f:
            json.dump(ids, f)
