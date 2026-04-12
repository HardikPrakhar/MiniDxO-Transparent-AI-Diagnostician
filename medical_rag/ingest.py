
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_medical_docs():
    urls = [
        "https://www.mayoclinic.org/diseases-conditions/flu/symptoms-causes/syc-20351719",
        "https://www.mayoclinic.org/diseases-conditions/diabetes/symptoms-causes/syc-20371444",
        "https://www.nhlbi.nih.gov/health/heart-attack",
        "https://www.nhlbi.nih.gov/health/asthma",
    ]

    loader = WebBaseLoader(urls)
    docs = loader.load()

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    split_docs = splitter.split_documents(docs)

    return split_docs


if __name__ == "__main__":
    docs = load_medical_docs()
    print(f"Loaded {len(docs)} chunks")