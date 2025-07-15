import os
import json
from datetime import datetime
from langchain.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
import uuid

VECTOR_STORAGE_DIR = './VectorStorage'
VECTOR_DIM = 384  # all-MiniLM-L6-v2 output dim
VECTOR_HISTORY_FILE = os.path.join(VECTOR_STORAGE_DIR, "history.json")
VECTOR_FAISS_FILE = os.path.join(VECTOR_STORAGE_DIR, "faiss_index.pkl")

class ProposalHistoryEntry:
    def __init__(self, date_time, job_text, proposal, comments, response_review, proposal_id=None):
        self.date_time = date_time
        self.job_text = job_text
        self.proposal = proposal
        self.comments = comments
        self.response_review = response_review
        self.proposal_id = proposal_id or str(uuid.uuid4())
    def to_dict(self):
        return {
            'date_time': self.date_time,
            'job_text': self.job_text,
            'proposal': self.proposal,
            'comments': self.comments,
            'response_review': self.response_review,
            'proposal_id': self.proposal_id
        }
    @staticmethod
    def from_dict(d):
        return ProposalHistoryEntry(
            d['date_time'], d['job_text'], d['proposal'], d['comments'], d['response_review'], d.get('proposal_id')
        )

# --- Save all proposals in one file ---
def save_proposal_history(entry: ProposalHistoryEntry):
    if not os.path.exists(VECTOR_STORAGE_DIR):
        os.makedirs(VECTOR_STORAGE_DIR)
    entries = []
    if os.path.exists(VECTOR_HISTORY_FILE):
        with open(VECTOR_HISTORY_FILE, 'r', encoding='utf-8') as f:
            try:
                entries = json.load(f)
            except Exception:
                entries = []
    # Remove any existing entry with same proposal_id
    entries = [e for e in entries if e.get('proposal_id') != entry.proposal_id]
    entries.append(entry.to_dict())
    with open(VECTOR_HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)
    # Update FAISS index
    update_faiss_index(entries)

# --- Get all entries from single file ---
def get_all_history_entries():
    entries = []
    if os.path.exists(VECTOR_HISTORY_FILE):
        with open(VECTOR_HISTORY_FILE, 'r', encoding='utf-8') as f:
            try:
                entries = json.load(f)
            except Exception:
                entries = []
    return [ProposalHistoryEntry.from_dict(e) for e in entries]

def get_st_embeddings():
    return SentenceTransformer('all-MiniLM-L6-v2')

# --- Persist FAISS index ---
def update_faiss_index(entries):
    model = get_st_embeddings()
    texts = []
    vectors = []
    metadatas = []
    for e in entries:
        try:
            text = e['job_text'] + ' ' + e['proposal'] + ' ' + (e.get('comments') or '')
            vec = model.encode([text])[0]
            if isinstance(vec, (list, tuple)) or hasattr(vec, 'shape'):
                if hasattr(vec, 'shape') and vec.shape[0] == VECTOR_DIM:
                    texts.append(text)
                    vectors.append(vec)
                    metadatas.append(e)
                elif isinstance(vec, (list, tuple)) and len(vec) == VECTOR_DIM:
                    texts.append(text)
                    vectors.append(vec)
                    metadatas.append(e)
        except Exception:
            continue
    if not texts or not vectors:
        return
    text_embeddings = list(zip(texts, vectors))
    db = FAISS.from_embeddings(
        text_embeddings=text_embeddings,
        embedding=model,
        metadatas=metadatas
    )
    db.save_local(VECTOR_FAISS_FILE)

# --- Load FAISS index ---
def load_faiss_index():
    if not os.path.exists(VECTOR_FAISS_FILE):
        return None
    model = get_st_embeddings()
    db = FAISS.load_local(VECTOR_FAISS_FILE, model)
    return db

# --- Retrieve similar history ---
def retrieve_similar_history(job_text, top_k=3):
    db = load_faiss_index()
    if db is None:
        return []
    model = get_st_embeddings()
    query_vec = model.encode([job_text])[0]
    docs = db.similarity_search_by_vector(query_vec, k=top_k)
    results = []
    for doc in docs:
        meta = doc.metadata
        results.append(ProposalHistoryEntry.from_dict(meta))
    return results

# --- Update entry by proposal_id ---
def update_proposal_history_by_id(proposal_id, comments, response_review):
    if not os.path.exists(VECTOR_HISTORY_FILE):
        return False
    with open(VECTOR_HISTORY_FILE, 'r', encoding='utf-8') as f:
        try:
            entries = json.load(f)
        except Exception:
            entries = []
    updated = False
    for e in entries:
        if e.get('proposal_id') == proposal_id:
            e['comments'] = comments
            e['response_review'] = response_review
            updated = True
    if updated:
        with open(VECTOR_HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(entries, f, ensure_ascii=False, indent=2)
        update_faiss_index(entries)
        return True
    return False
