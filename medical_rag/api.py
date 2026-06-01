from fastapi import FastAPI
from pydantic import BaseModel

from doctor_agent import generate_response, memory

app = FastAPI()

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
def chat(req: ChatRequest):
    response = generate_response(req.message)
    return {"response": response}

@app.get("/history")
def history():
    return memory.load_memory_variables({})

@app.post("/reset")
def reset():
    memory.clear()
    return {"status": "cleared"}