# process_existing.py
import os
from app import DOCUMENTS_DIR, process_pdf, save_artifacts, MODEL_DIMENSIONS, model
import faiss
import pickle

# Reinitialize the FAISS index and the documents list
index = faiss.IndexFlatL2(MODEL_DIMENSIONS)
documents = []

print("Starting initial processing of all PDFs in", DOCUMENTS_DIR)

if os.path.exists(DOCUMENTS_DIR):
    for filename in os.listdir(DOCUMENTS_DIR):
        if filename.lower().endswith('.pdf'):
            pdf_path = os.path.join(DOCUMENTS_DIR, filename)
            print("Processing:", filename)
            # process_pdf should update the global index and documents;
            # you might need to adjust your code so that process_pdf updates the same index/documents used by your app.
            process_pdf(pdf_path, filename)

save_artifacts()
print("Initial processing complete.")
