import os
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from flask import Flask, request, render_template, redirect, send_from_directory
from pypdf import PdfReader

app = Flask(__name__)

# Configuration
# Set DOCUMENTS_DIR to the container path where the host folder is mounted
DOCUMENTS_DIR = "documents"
INDEX_FILE = "faiss_index.index"
METADATA_FILE = "metadata.pkl"

MODEL_NAME = "all-mpnet-base-v2"
MODEL_DIMENSIONS = 768

MIN_THRESHOLD = 1.75
MOST_RELEVANT_THRESHOLD = 1.25

# Global variables for the index and document metadata
model = SentenceTransformer(MODEL_NAME)

# Initialize index and documents list
if os.path.exists(INDEX_FILE):
    index = faiss.read_index(INDEX_FILE)
else:
    index = faiss.IndexFlatL2(MODEL_DIMENSIONS)

if os.path.exists(METADATA_FILE):
    with open(METADATA_FILE, "rb") as f:
        documents = pickle.load(f)
else:
    documents = []


def save_artifacts():
    faiss.write_index(index, INDEX_FILE)
    with open(METADATA_FILE, "wb") as f:
        pickle.dump(documents, f)


def process_pdf(pdf_path, pdf_name):
    try:
        reader = PdfReader(pdf_path)
    except Exception as e:
        print(f"Error reading {pdf_name}: {e}")
        return 0

    new_entries = []
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        if text and text.strip():  # Ensure page has nonempty text
            new_entries.append((pdf_name, page_num, text))

    if new_entries:
        # Encode and add pages to the FAISS index
        embeddings = model.encode(
            [entry[2] for entry in new_entries],
            normalize_embeddings=True
        )
        index.add(np.array(embeddings).astype('float32'))
        documents.extend(new_entries)

    return len(new_entries)


@app.route('/')
def home():
    try:
        pdf_count = len(
            [f for f in os.listdir(DOCUMENTS_DIR) if f.lower().endswith('.pdf')]
        )
    except FileNotFoundError:
        pdf_count = 0
    # Render search.html with only the search bar
    return render_template('search.html', document_count=pdf_count)


@app.route('/upload-page')
def upload_page():
    try:
        pdf_count = len(
            [f for f in os.listdir(DOCUMENTS_DIR) if f.lower().endswith('.pdf')]
        )
    except FileNotFoundError:
        pdf_count = 0
    return render_template('upload.html', document_count=pdf_count)


@app.route('/process_all', methods=['POST'])
def process_all():
    global index, documents
    # Reset index and metadata
    index = faiss.IndexFlatL2(MODEL_DIMENSIONS)
    documents = []

    for filename in os.listdir(DOCUMENTS_DIR):
        if filename.lower().endswith('.pdf'):
            pdf_path = os.path.join(DOCUMENTS_DIR, filename)
            process_pdf(pdf_path, filename)

    save_artifacts()
    return redirect('/')


@app.route('/upload', methods=['POST'])
def upload_document():
    if 'pdf' not in request.files:
        return "No PDF uploaded", 400

    pdf_file = request.files['pdf']
    if pdf_file.filename == '':
        return "No selected file", 400

    # Ensure the documents directory exists
    if not os.path.exists(DOCUMENTS_DIR):
        os.makedirs(DOCUMENTS_DIR)

    pdf_path = os.path.join(DOCUMENTS_DIR, pdf_file.filename)
    pdf_file.save(pdf_path)

    pages_processed = process_pdf(pdf_path, pdf_file.filename)
    save_artifacts()

    return f"PDF processed successfully. Added {pages_processed} pages.", 200


def group_results(results):
    # Group results by PDF and filter out pages above the threshold.
    grouped = {}
    for res in results:
        if res["score"] > MIN_THRESHOLD:
            continue  # omit pages with poor scores
        pdf_name = res["pdf"]
        grouped.setdefault(pdf_name, []).append(res)

    # For each PDF, sort pages and group continuous page numbers
    pdf_groups = {}
    for pdf, pages in grouped.items():
        if not pages:
            continue  # Skip if no valid pages remain
        pages = sorted(pages, key=lambda x: x["page"])
        ranges = []
        current_range = [pages[0]]
        for page in pages[1:]:
            if page["page"] == current_range[-1]["page"] + 1:
                current_range.append(page)
            else:
                ranges.append(current_range)
                current_range = [page]
        ranges.append(current_range)  # add the final group

        clubbed = []
        for group in ranges:
            start_page = group[0]["page"]
            end_page = group[-1]["page"]
            min_score = min(item["score"] for item in group)
            classification = (
                "most relevant" if min_score <= MOST_RELEVANT_THRESHOLD else "maybe"
            )
            link = f"/documents/{pdf}#page={start_page}"
            clubbed.append({
                "range": f"{start_page}-{end_page}" if start_page != end_page else f"{start_page}",
                "score": min_score,
                "classification": classification,
                "link": link
            })
        pdf_groups[pdf] = clubbed
    return pdf_groups


@app.route('/search')
def search():
    query = request.args.get('q')
    try:
        pdf_count = len(
            [f for f in os.listdir(DOCUMENTS_DIR) if f.lower().endswith('.pdf')]
        )
    except FileNotFoundError:
        pdf_count = 0

    if not query:
        return redirect('/')
    if index.ntotal == 0:
        return render_template('search.html', document_count=pdf_count, query=query, grouped_results=None)

    query_embedding = model.encode([query], normalize_embeddings=True)
    query_vector = np.array(query_embedding).astype('float32')
    k = min(10, index.ntotal)
    distances, indices = index.search(query_vector, k=k)

    raw_results = []
    for i, idx in enumerate(indices[0]):
        if 0 <= idx < len(documents):
            pdf_name, page_num, text = documents[idx]
            raw_results.append({
                'pdf': pdf_name,
                'page': page_num,
                'score': float(distances[0][i]),
                'text': text[:200] + '...',
                'url': f"/documents/{pdf_name}#page={page_num}"
            })

    # Group results by PDF and club consecutive pages
    grouped_results = group_results(raw_results)
    pdf_overall = {
        pdf: max(g['score'] for g in groups)
        for pdf, groups in grouped_results.items()
    } if grouped_results else {}

    return render_template(
        'search.html',
        document_count=pdf_count,
        query=query,
        grouped_results=grouped_results,
        pdf_overall=pdf_overall
    )


@app.route('/documents/<filename>')
def serve_pdf(filename):
    return send_from_directory(DOCUMENTS_DIR, filename)


if __name__ == '__main__':
    # Ensure the persistent documents directory exists
    if not os.path.exists(DOCUMENTS_DIR):
        os.makedirs(DOCUMENTS_DIR)

    # === INITIAL PROCESSING ON STARTUP ===
    # This block will process all PDFs found in /documents (persistent storage) on deployment
    print("Starting initial processing of all PDFs in", DOCUMENTS_DIR)
    # Reset index and documents list on each deployment startup
    index = faiss.IndexFlatL2(MODEL_DIMENSIONS)
    documents = []
    for filename in os.listdir(DOCUMENTS_DIR):
        if filename.lower().endswith('.pdf'):
            pdf_path = os.path.join(DOCUMENTS_DIR, filename)
            print("Processing:", filename)
            process_pdf(pdf_path, filename)
    save_artifacts()
    print("Initial processing complete. Indexed", len(documents), "pages.")

    app.run(host='0.0.0.0', port=3000)
