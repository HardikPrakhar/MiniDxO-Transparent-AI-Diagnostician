from retriever import get_retriever

retriever = get_retriever()

query = "I have chest pain and shortness of breath"
docs = retriever.invoke(query)

for i, doc in enumerate(docs):
    print(f"\n--- Doc {i+1} ---")
    print(doc.page_content[:500])