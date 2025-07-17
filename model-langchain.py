import os
import csv
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np

EMBEDDING_MODEL = 'all-MiniLM-L6-v2'

DATA_CSV = './upwork-data/weblink-for-upwork-proposal.csv'
UPWORK_DATA_CSV = './data/upwork-portfolios.csv'
FIGMA_DATA_CSV = './data/figma-portfolios.csv'

EMBEDDINGS_FILE = './embeddings/website_embeddings_2.pkl'
UPWORK_EMBEDDINGS_FILE = './embeddings/upwork_embeddings.pkl'
FIGMA_EMBEDDINGS_FILE = './embeddings/figma_embeddings.pkl'

# Load website data
def load_website_data():
    with open(DATA_CSV, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        print("Raw CSV Headers Detected:", reader.fieldnames)

        data = []
        for row in reader:
            cleaned_row = {k.strip(): v.strip() for k, v in row.items()}
            data.append(cleaned_row)

        return data



def load_upwork_data():
    upwork_websites = []
    try:
        with open(UPWORK_DATA_CSV, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                upwork_websites.append(row)
    except UnicodeDecodeError:
        with open(UPWORK_DATA_CSV, 'r', encoding='latin1') as f:
            reader = csv.DictReader(f)
            for row in reader:
                upwork_websites.append(row)
    return upwork_websites


def load_figma_data():
    figmas = []
    with open(FIGMA_DATA_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            figmas.append(row)
    return figmas

def create_and_store_embeddings():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    websites = load_website_data()

    texts = [
        f"""Link: {w['Website URL']}
        Category: {w['Category']}
        Sub Category: {w.get('sub category', '')}
        Niche: {w.get('Niche', '')}
        Priority: {w.get('Priority', '')}
        Description: {w['Website Description']}
        Keywords: {w['Keywords']}
        Tech Stack: {w['Tech Stack']}"""
        for w in websites
    ]

    embeddings = model.encode(texts, show_progress_bar=True)
    
    os.makedirs(os.path.dirname(EMBEDDINGS_FILE), exist_ok=True)
    with open(EMBEDDINGS_FILE, 'wb') as f:
        pickle.dump({'embeddings': embeddings, 'websites': websites, 'texts': texts}, f)

    print(f"Embeddings saved to {EMBEDDINGS_FILE}")



def create_and_store_embeddings_upwork():
    model = SentenceTransformer(EMBEDDING_MODEL)
    websites_upwork = load_upwork_data()
    texts = [
        f"Link: {w['Website Link']}\nProject Name: {w.get('Project Name', '')}\nCategory: {w['Category']}\nDescription: {w['Description']}\nKeywords: {w['Keywords']}\nTech Stack: {w.get('Tech Stack', '')}"
        for w in websites_upwork
    ]
    embeddings = model.encode(texts, show_progress_bar=True)
    os.makedirs(os.path.dirname(UPWORK_EMBEDDINGS_FILE), exist_ok=True)
    with open(UPWORK_EMBEDDINGS_FILE, 'wb') as f:
        pickle.dump({'embeddings': embeddings, 'websites': websites_upwork, 'texts': texts}, f)
    print(f"Embeddings saved to {UPWORK_EMBEDDINGS_FILE}")


def create_and_store_embeddings_figma():
    model = SentenceTransformer(EMBEDDING_MODEL)
    figmas = load_figma_data()
    texts = [
        f"Link: {f['Figma Link']}\nProject Name: {f.get('Project Name', '')}\nCategory: {f['Category']}\nProto-type: {f.get('Proto-type', '')}\nWebsite Link: {f.get('Website Link', '')}\nLanguage: {f.get('Language', '')}"
        for f in figmas
    ]
    embeddings = model.encode(texts, show_progress_bar=True)
    os.makedirs(os.path.dirname(FIGMA_EMBEDDINGS_FILE), exist_ok=True)
    with open(FIGMA_EMBEDDINGS_FILE, 'wb') as f:
        pickle.dump({'embeddings': embeddings, 'figmas': figmas, 'texts': texts}, f)
    print(f"Embeddings saved to {FIGMA_EMBEDDINGS_FILE}")


def test_embedding_search(query, top_k=3):
    model = SentenceTransformer(EMBEDDING_MODEL)
    with open(EMBEDDINGS_FILE, 'rb') as f:
        data = pickle.load(f)
    embeddings = data['embeddings']
    websites = data['websites']
    texts = data['texts']
    query_emb = model.encode([query])[0]
    similarities = np.dot(embeddings, query_emb) / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_emb))
    top_indices = similarities.argsort()[-top_k:][::-1]
    print(f"Top {top_k} results for query: '{query}'\n")
    for i in top_indices:
        print(f"Score: {similarities[i]:.3f}")
        print(texts[i])
        print('-'*60)

def test_embedding_search_upwork(query, top_k=3):
    model = SentenceTransformer(EMBEDDING_MODEL)
    with open(UPWORK_EMBEDDINGS_FILE, 'rb') as f:
        data = pickle.load(f)
    embeddings = data['embeddings']
    websites = data['websites']
    texts = data['texts']
    query_emb = model.encode([query])[0]
    similarities = np.dot(embeddings, query_emb) / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_emb))
    top_indices = similarities.argsort()[-top_k:][::-1]
    print(f"Top {top_k} results for query: '{query}'\n")
    for i in top_indices:
        print(f"Score: {similarities[i]:.3f}")
        print(texts[i])
        print('-'*60)


def test_embedding_search_figma(query, top_k=3):
    model = SentenceTransformer(EMBEDDING_MODEL)
    with open(FIGMA_EMBEDDINGS_FILE, 'rb') as f:
        data = pickle.load(f)
    embeddings = data['embeddings']
    figmas = data['figmas']
    texts = data['texts']
    query_emb = model.encode([query])[0]
    similarities = np.dot(embeddings, query_emb) / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_emb))
    top_indices = similarities.argsort()[-top_k:][::-1]
    print(f"Top {top_k} results for query: '{query}'\n")
    for i in top_indices:
        print(f"Score: {similarities[i]:.3f}")
        print(texts[i])
        print('-'*60)

if __name__ == '__main__':
    create_and_store_embeddings()
    # test_embedding_search("We’re a dental clinic in California needing a brand-new site with appointment scheduling, services overview, patient forms download, testimonials, and blog integration")
    
    # create_and_store_embeddings_figma()
