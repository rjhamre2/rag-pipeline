from fastapi import FastAPI
from pydantic import BaseModel
from rag_chain import build_qa_chain
import time

app = FastAPI()
qa_chain = build_qa_chain()

class Question(BaseModel):
    question: str

@app.post("/ask")
async def ask_question(q: Question):
    start_time = time.time()
    response = qa_chain.run(q.question)
    end_time = time.time()
    return {"answer": response, "processing_time": end_time - start_time}