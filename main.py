from fastapi import FastAPI
from pydantic import BaseModel
from rag_chain import build_qa_chain
from embedding_generator import get_embedding_model
from get_vectorstore import get_vector_store
from model_list import vectorDB_Path
import time
from fastapi.middleware.cors import CORSMiddleware
import faiss
import numpy as np

app = FastAPI()
#qa_chain = build_qa_chain()

class Question(BaseModel):
    question: str

origins = [
    "http://localhost:3000",  # Example: React frontend running locally
    "https://nimbleai.in",  # Example: Deployed frontend
]

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # List of allowed origins
    allow_credentials=True,  # Allow cookies or authentication headers
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)


vector_store = "faiss"
embedding_name = "nomic-embed-text"
persist_directory = f"{vectorDB_Path}{embedding_name}_{vector_store}_db"

vectorstore = get_vector_store(embedding_name=embedding_name,
                                store_name=vector_store, 
                                persist_directory=persist_directory)
docs = vectorstore.docstore._dict
index = vectorstore.index
num_vectors = index.ntotal
print(f"faiss  dimensions: {index.d}, Number of vectors: {num_vectors}")
#print(f"query_vector  dimensions: {query_vector.shape}")
all_vectors = index.reconstruct_n(0, num_vectors)  # Shape: [num_vectors, dims]


@app.post("/ask")
async def ask_question(q: Question):
    start_time = time.time()
#    response = qa_chain.run(q.question)
    question = q.question
    embeddings = get_embedding_model(embedding_name)
    query_embedding = embeddings.embed_query(question)
    # Convert to 2D float32 array
    query_vector = np.array([query_embedding], dtype="float32")  # shape (1, d)

    # Normalize in-place
    faiss.normalize_L2(query_vector)
    k = 5  # number of nearest neighbors
    distances, indices = vectorstore.index.search(query_vector, k)
    doc_id = vectorstore.index_to_docstore_id[indices[0][0]]
    # Step 2: Fetch the document from the docstore
    document = docs[doc_id]
    end_time = time.time()
    return {"answer": document, "processing_time": end_time - start_time}
