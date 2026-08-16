import uuid

from fastapi import FastAPI, Header
from pydantic import BaseModel

from doctor_agent import clear_memory, generate_response, get_memory

app = FastAPI()


class ChatRequest(BaseModel):
    message: str


@app.post("/chat")
def chat(req: ChatRequest, x_session_id: str = Header(default=None)):
    session_id = x_session_id or str(uuid.uuid4())
    response = generate_response(req.message, session_id=session_id)
    
    return {"response": response, "session_id": session_id}


@app.get("/history")
def history(x_session_id: str = Header(...)):
    return get_memory(x_session_id).load_memory_variables({})


@app.post("/reset")
def reset(x_session_id: str = Header(...)):
    clear_memory(x_session_id)
    return {"status": "cleared"}
