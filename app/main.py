from fastapi import FastAPI
from pydantic import BaseModel

from app.nimbot_core import build_qa_chain

app = FastAPI()
qa_chain = build_qa_chain()

class Query(BaseModel):
    question: str

@app.post("/ask")
def ask_nimbot(query: Query):
    response = qa_chain.invoke({"query": query.question})
    return {
        "answer": response["result"],
        "sources": [doc.metadata["source"] for doc in response["source_documents"]]
    }

@app.post("/refresh")
def refresh_nimbot():
    global qa_chain
    qa_chain = build_qa_chain()
    return {"message": "🧠 Nimbot has been refreshed! 🔁"}

@app.get("/")
def root():
    return {"message": "🤖 Nimbot is alive. POST a query to /ask. POST /refresh to trigger a refresh from experimenter.info"}


