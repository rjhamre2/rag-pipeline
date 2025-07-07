from typing import Literal, Union
from langchain_community.vectorstores import FAISS, Chroma, LanceDB, Weaviate
#from langchain_community.vectorstores import Weaviate #docker setup needed
from langchain_community.embeddings import OllamaEmbeddings, HuggingFaceEmbeddings
#import weaviate #docker setup needed
from langchain_core.vectorstores import VectorStore
from embedding_generator import get_embedding_model
from langchain.schema import Document
from model_list import vectorDB_Path, chunk_sizes, chunk_overlaps, embeddings_models, vector_store_type
from faqs import faq_data
import faiss
from sklearn.preprocessing import normalize
import time
import numpy as np

def get_vectorstore(
    store_name: Literal["faiss", "chroma", "lancedb", "weaviate"],
    embedding_name: str,
    documents: list = None,
    persist_directory: str = None,
    **kwargs
) -> Union[FAISS, Chroma, LanceDB]: #include Weaviate if needed
    """
    Initialize and return a vector store for RAG applications.
    
    Args:
        store_name: Type of vector store ("faiss", "chroma", "lancedb", "weaviate")
        embedding_name: Name of the embedding model (must match your embeddings_models)
        documents: List of documents to initialize the store (optional)
        persist_directory: Directory to persist the store (required for Chroma/LanceDB)
        **kwargs: Additional store-specific arguments
        
    Returns:
        Initialized vector store ready for RAG
        
    Examples:
        >>> # FAISS (in-memory)
        >>> vectorstore = get_vectorstore("faiss", "bge-small-en", documents=texts)
        
        >>> # Chroma (persistent)
        >>> vectorstore = get_vectorstore("chroma", "nomic-embed-text", 
        ...                             documents=texts, persist_directory="./chroma_db")
    """
    # Get embedding model
    embeddings = get_embedding_model(embedding_name)
    
    # Initialize the requested vector store
    if store_name == "faiss":
        if documents is None:
            raise ValueError("FAISS requires documents for initialization")
        vectorstore = FAISS.from_documents(documents, embeddings)
        #vectorstore.save_local(f"{vectorDB_Path}{embedding_name}_faiss_index/")
        faiss_index = vectorstore.index
        # Reconstruct all vectors
        vectors = np.array([faiss_index.reconstruct(i) for i in range(faiss_index.ntotal)]).astype("float32")

# Normalize in-place (no return)
        faiss.normalize_L2(vectors)
        # Create a new FAISS index with normalized vectors
        dimension = faiss_index.d
        new_index = faiss.IndexFlatL2(dimension)
        new_index.add(vectors)

        # Replace the old index in vectorstore
        vectorstore.index = new_index
        vectorstore.save_local(persist_directory)
        return vectorstore

    elif store_name == "chroma":
        if not persist_directory:
            raise ValueError("Chroma requires a persist_directory")
        return Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=persist_directory,
            **kwargs
        )
        
    elif store_name == "lancedb":
        if not persist_directory:
            raise ValueError("LanceDB requires a persist_directory")
        db = lancedb.connect(persist_directory)
        return LanceDB.from_documents(
            documents=documents,
            embedding=embeddings,
            connection=db,
            **kwargs
        )   
    else:
        raise ValueError(f"Unsupported store: {store_name}. Choose from: faiss, chroma, lancedb, weaviate")


langchain_docs = [
    Document(
        page_content=question,  # using the question as page_content
        metadata={
            "answer": answer,
            "source": "FAQ",
            "category": "General"
        }
    )
    for question, answer in faq_data.items()
]
# Step 3: Generate embeddings
embedding_model = "nomic-embed-text"
vector_store = "faiss"

persist_directory = f"{vectorDB_Path}{embedding_model}_{vector_store}_db"

vectorstore = get_vectorstore(
    store_name=vector_store,
    embedding_name=embedding_model,
    documents=langchain_docs,
    persist_directory=persist_directory
    )


                
