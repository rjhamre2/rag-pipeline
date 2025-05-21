from typing import Optional, List, Union
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS, Chroma, LanceDB
from langchain_community.embeddings import OllamaEmbeddings, HuggingFaceEmbeddings
import lancedb
from embedding_generator import get_embedding_model
def get_vector_store(
    store_name: str,
    embedding_name: str,
    persist_directory: str,
    documents: Optional[List[Document]] = None,
    **kwargs
) -> Union[FAISS, Chroma, LanceDB]:
    """
    Get or create a vector store with specified embedding model and storage type.
    
    Args:
        store_name: Type of vector store ("faiss", "chroma", "lancedb")
        embedding_name: Name of embedding model from your embeddings_models
        persist_directory: Directory to save/load the vector store
        documents: Documents to create new store (optional for loading existing)
        **kwargs: Additional arguments for specific vector stores
        
    Returns:
        Initialized vector store ready for retrieval
    """
    # Get embedding model
    embeddings = get_embedding_model(embedding_name)
    
    # Handle each store type
    if store_name == "faiss":
        try:
            # Try to load existing index
            vectorstore = FAISS.load_local(
                folder_path=persist_directory,
                embeddings=embeddings,
                allow_dangerous_deserialization=True  # Required for FAISS
            )
            print(f"Loaded existing FAISS index from {persist_directory}")
            return vectorstore
        except:
            if documents is None:
                raise ValueError("FAISS requires documents when creating new index")
            print("Creating new FAISS index")
            vectorstore = FAISS.from_documents(documents, embeddings)
            vectorstore.save_local(persist_directory)
            return vectorstore
            
    elif store_name == "chroma":
        # Chroma handles persistence automatically
        if documents is not None:
            print("Creating/updating Chroma collection")
            return Chroma.from_documents(
                documents=documents,
                embedding=embeddings,
                persist_directory=persist_directory,
                **kwargs
            )
        else:
            print("Loading existing Chroma collection")
            return Chroma(
                persist_directory=persist_directory,
                embedding_function=embeddings,
                **kwargs
            )
            
    elif store_name == "lancedb":
        db = lancedb.connect(persist_directory)
        
        table_name = kwargs.get("table_name", "vectorstore")
        #print(f"table name is {table_name}")  # Check if your table exists
        #print(f"Available tables in {persist_directory}: {db.table_names()}")
        table = db.open_table(table_name)
        #print(f"table is {table}")
        if documents is not None:
            print("Creating/updating LanceDB table")
            return LanceDB.from_documents(
                documents=documents,
                embedding=embeddings,
                connection=db,
                table_name=table_name,
                **kwargs
            )
        else:
            print("Loading existing LanceDB table")
            return LanceDB(
                connection=db,
                embedding=embeddings,
                table=table,
                **kwargs
            )
            
    else:
        raise ValueError(f"Unsupported store: {store_name}")