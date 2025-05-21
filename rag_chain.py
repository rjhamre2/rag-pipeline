from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from get_vectorstore import get_vector_store
from model_list import lightweight_ollama_models
from model_list import vectorDB_Path, chunk_sizes, chunk_overlaps, embeddings_models, vector_store_type

def build_qa_chain():
    chunk_size = chunk_sizes[0]
    chunk_overlap = chunk_overlaps[1]
    embedding_model = "all-minilm-l6-v2"
    vector_store = vector_store_type[0]

    llmModel = lightweight_ollama_models[0]
    persist_directory = f"{vectorDB_Path}{chunk_size}_{chunk_overlap}_{embedding_model}_{vector_store}_db"
    llm = Ollama(model=llmModel['name'],
            temperature=0,
            num_ctx =2048,
            num_thread=8,  # Match to CPU cores
            repeat_penalty=1.1,
            top_k=40,
            top_p=0.9,)

    vectorstore = get_vector_store(embedding_name=embedding_model,
                                store_name=vector_store, 
                                persist_directory=persist_directory)
    qa_chain = RetrievalQA.from_chain_type(
                                llm=llm,
                                retriever=vectorstore.as_retriever(search_kwargs={"k": 1}),
                                chain_type="stuff",
                                return_source_documents=False)

    return qa_chain