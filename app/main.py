from fastapi import FastAPI
from pydantic import BaseModel

from app.nimbot_core import build_qa_chains, invoke_with_fallback

app = FastAPI()
build_qa_chains()

class Query(BaseModel):
    question: str

@app.post("/ask")
def ask_nimbot(query: Query):
    response = invoke_with_fallback(query.question)
    return {
        "answer": response["result"],
        "sources": [doc.metadata["source"] for doc in response["source_documents"]]
    }

@app.post("/refresh")
def refresh_nimbot():
    build_qa_chains()
    return {"message": "🧠 Nimbot has been refreshed! 🔁"}

@app.get("/")
def root():
    return {"message": "🤖 Nimbot is alive. POST a query to /ask. POST /refresh to trigger a refresh from experimenter.info"}


