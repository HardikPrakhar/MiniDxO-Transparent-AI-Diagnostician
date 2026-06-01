from retriever import get_retriever
from dotenv import load_dotenv

load_dotenv()

import os

from langchain_openai import ChatOpenAI
from langchain_classic.memory import ConversationBufferMemory

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

print(type(OPENROUTER_API_KEY))
print(repr(OPENROUTER_API_KEY))

# Memory
memory = ConversationBufferMemory(
    memory_key="history",
    return_messages=False
)

llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    model="meta-llama/llama-3-8b-instruct",
    temperature=0.3,
    default_headers={
        "HTTP-Referer": "http://localhost:8000",
        "X-Title": "MiniDxO"
    }
)

retriever = get_retriever()

SYSTEM_PROMPT = """
You are an AI medical assistant simulating a doctor.

Rules:
- Ask follow-up questions if needed
- Use past conversation
- Use provided medical context
- Do NOT give definitive diagnosis
- Always include sources when explaining
- Be cautious and safe

Output format:
1. Follow-up Questions (if needed)
2. Possible Conditions
3. Explanation (with sources)
4. Advice
"""

def generate_response(user_query: str):
    docs = retriever.invoke(user_query)

    context = "\n\n".join(
        [
            f"{doc.page_content}\n(Source: {doc.metadata.get('source', 'Unknown')})"
            for doc in docs
        ]
    )

    history = memory.load_memory_variables({}).get("history", "")

    final_prompt = f"""
{SYSTEM_PROMPT}

Chat History:
{history}

Medical Context:
{context}

User Symptoms:
{user_query}
"""

    response = llm.invoke(final_prompt)

    memory.save_context(
        {"input": user_query},
        {"output": response.content}
    )

    return response.content