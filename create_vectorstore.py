from typing import Literal, Union
from langchain_community.vectorstores import FAISS, Chroma, LanceDB, Weaviate
#from langchain_community.vectorstores import Weaviate #docker setup needed
from langchain_community.embeddings import OllamaEmbeddings, HuggingFaceEmbeddings
import lancedb
#import weaviate #docker setup needed
from langchain_core.vectorstores import VectorStore
from embedding_generator import get_embedding_model
from langchain.schema import Document
from model_list import vectorDB_Path, chunk_sizes, chunk_overlaps, embeddings_models, vector_store_type

import time
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
    

from DocumentChunker import DocumentChunker
from langchain_community.document_loaders import TextLoader

loader = TextLoader("faqs.txt",encoding="utf-8")
documents = loader.load()

chunk_size = chunk_sizes[0]
chunk_overlap = chunk_overlaps[1]
embedding_model = "all-minilm-l6-v2"
vector_store = vector_store_type[0]
# Step 2: Chunk documents
chunker = DocumentChunker(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    separators=["\n\n", "\nChapter", "(?<=\. )", " "]
    )
chunks = chunker.chunk_documents(
    documents,
    extra_metadata={"source1": "myntra website"}
)
langchain_docs = [
    Document(
        page_content=chunk["text"],
        metadata=chunk["metadata"]  # Preserve existing metadata
        )for chunk in chunks ]

# Step 3: Generate embeddings

persist_directory = f"{vectorDB_Path}{chunk_size}_{chunk_overlap}_{embedding_model}_{vector_store}_db"

vectorstore = get_vectorstore(
    store_name=vector_store,
    embedding_name=embedding_model,
    documents=langchain_docs,
    persist_directory=persist_directory
    )


                