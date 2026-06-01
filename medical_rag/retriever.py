
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def get_retriever():
    embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)
    db = FAISS.load_local(
        "../medical_db",
        embeddings,
        allow_dangerous_deserialization=True
    )

    return db.as_retriever(search_kwargs={"k": 3})