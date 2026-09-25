from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from agent_lo import run
import uvicorn

app = FastAPI(title="Research Assistant API")


class ChatRequest(BaseModel):
    query: str
    thread_id: str


class ChatResponse(BaseModel):
    answer: str
    plan: list[str]
    draft: str
    critique: str


@app.get("/")
def read_root():
    return {"status": "Research Assistant is running 🚀"}


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    try:
        return ChatResponse(**run(request.query, request.thread_id))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)