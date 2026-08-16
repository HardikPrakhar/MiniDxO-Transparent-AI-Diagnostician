import os
from typing import List, Optional, TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from retriever import get_retriever

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    model="openrouter/free",
    temperature=0.3,
    default_headers={
        "HTTP-Referer": "http://localhost:8000",
        "X-Title": "MiniDxO",
    },
)

retriever = get_retriever()


# ---------------------------------------------------------------------------
# Shared state passed between agents
# ---------------------------------------------------------------------------
class AgentState(TypedDict, total=False):
    user_query: str
    history: str
    needs_more_info: bool
    follow_up_questions: Optional[str]
    retrieved_context: Optional[str]
    diagnosis_draft: Optional[str]
    final_response: Optional[str]


# ---------------------------------------------------------------------------
# Agent 1: Intake — decides whether there's enough information to proceed
# ---------------------------------------------------------------------------
def intake_agent(state: AgentState) -> AgentState:
    prompt = f"""You are the intake step of a medical assistant.
Given the chat history and the latest message, decide if there is enough
information to reason about possible conditions, or if follow-up questions
are needed first.

Chat History:
{state.get('history', '')}

Latest message:
{state['user_query']}

Reply in exactly one of these two formats, nothing else:
NEEDS_INFO: <one or two concise follow-up questions>
SUFFICIENT
"""
    result = llm.invoke(prompt).content.strip()

    if result.upper().startswith("NEEDS_INFO"):
        questions = result.split(":", 1)[1].strip() if ":" in result else result
        return {**state, "needs_more_info": True, "follow_up_questions": questions}

    return {**state, "needs_more_info": False}


def route_after_intake(state: AgentState) -> str:
    return "end" if state.get("needs_more_info") else "retrieve"


# ---------------------------------------------------------------------------
# Agent 2: Retrieval — fetches grounding context from the vector store
# ---------------------------------------------------------------------------
def retrieval_agent(state: AgentState) -> AgentState:
    docs = retriever.invoke(state["user_query"])
    context = "\n\n".join(
        f"{doc.page_content}\n(Source: {doc.metadata.get('source', 'Unknown')})"
        for doc in docs
    )
    return {**state, "retrieved_context": context}


# ---------------------------------------------------------------------------
# Agent 3: Diagnosis — proposes possible conditions grounded in context
# ---------------------------------------------------------------------------
def diagnosis_agent(state: AgentState) -> AgentState:
    prompt = f"""You are the diagnosis-reasoning step of a medical assistant.
Using ONLY the medical context below, list plausible possible conditions for
these symptoms. Do NOT give a definitive diagnosis. Be cautious.

Medical Context:
{state.get('retrieved_context', '')}

Symptoms:
{state['user_query']}

List the possible conditions with a one-line rationale for each.
"""
    draft = llm.invoke(prompt).content
    return {**state, "diagnosis_draft": draft}


# ---------------------------------------------------------------------------
# Agent 4: Explanation — turns the draft into the final patient-facing reply
# ---------------------------------------------------------------------------
def explanation_agent(state: AgentState) -> AgentState:
    prompt = f"""You are the explanation step of a medical assistant. Turn the
draft below into a clear, patient-facing response using exactly these
sections:

1. Possible Conditions
2. Explanation (cite sources from the context)
3. Advice

Be cautious and safe. Do NOT give a definitive diagnosis.

Draft possible conditions:
{state.get('diagnosis_draft', '')}

Medical context (for sourcing):
{state.get('retrieved_context', '')}
"""
    final = llm.invoke(prompt).content
    return {**state, "final_response": final}


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------
def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("intake", intake_agent)
    graph.add_node("retrieve", retrieval_agent)
    graph.add_node("diagnose", diagnosis_agent)
    graph.add_node("explain", explanation_agent)

    graph.set_entry_point("intake")
    graph.add_conditional_edges(
        "intake", route_after_intake, {"end": END, "retrieve": "retrieve"}
    )
    graph.add_edge("retrieve", "diagnose")
    graph.add_edge("diagnose", "explain")
    graph.add_edge("explain", END)

    return graph.compile()


app_graph = build_graph()


def run_agent_graph(user_query: str, history: str = "") -> str:
    """Entry point used by doctor_agent.py. Runs the full agent graph and
    returns whatever the terminal node produced: either follow-up questions
    (if intake decided more info was needed) or the final formatted response.
    """
    result = app_graph.invoke({"user_query": user_query, "history": history})
    return result.get("final_response") or result.get("follow_up_questions") or ""
