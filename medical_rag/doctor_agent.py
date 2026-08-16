import logging

from langchain_classic.memory import ConversationBufferMemory

from agents import run_agent_graph

logger = logging.getLogger("minidxo")
logging.basicConfig(level=logging.INFO)


_sessions: dict[str, ConversationBufferMemory] = {}


def get_memory(session_id: str) -> ConversationBufferMemory:
    if session_id not in _sessions:
        logger.info("Creating new memory for session %s", session_id)
        _sessions[session_id] = ConversationBufferMemory(
            memory_key="history", return_messages=False
        )
    return _sessions[session_id]


def clear_memory(session_id: str) -> None:
    if session_id in _sessions:
        _sessions[session_id].clear()


def generate_response(user_query: str, session_id: str = "default") -> str:
    memory = get_memory(session_id)
    history = memory.load_memory_variables({}).get("history", "")

    try:
        response_text = run_agent_graph(user_query=user_query, history=history)
    except Exception:
        logger.exception("Agent graph failed for session %s", session_id)
        return (
            "Sorry, something went wrong while processing that. "
            "Please try again in a moment."
        )

    memory.save_context({"input": user_query}, {"output": response_text})
    return response_text
