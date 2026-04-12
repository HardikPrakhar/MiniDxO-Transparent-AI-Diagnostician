from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from ingest import load_medical_docs
import os

def create_vector_db():
    docs = load_medical_docs()

    embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

    db = FAISS.from_documents(docs, embeddings)

    # Save locally
    db.save_local("medical_db")

    print("Vector DB created and saved!")

if __name__ == "__main__":
    create_vector_db()