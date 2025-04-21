from fastapi import FastAPI
from pydantic import BaseModel
from rag_chain import build_qa_chain

app = FastAPI()
qa_chain = build_qa_chain()

class Question(BaseModel):
    question: str

@app.post("/ask")
async def ask_question(q: Question):
    response = qa_chain.run(q.question)
    return {"answer": response}