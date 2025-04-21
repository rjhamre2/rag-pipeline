from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

def build_qa_chain():
    # Load and split documents
    loader = TextLoader("faqs.txt")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    # Create embeddings and vector store
    embedding = OllamaEmbeddings(model="llama2:13b")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory="./chroma_db"
    )

    retriever = vectorstore.as_retriever()

    # Connect retriever to LLM
    llm = Ollama(model="llama2:13b")

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=False
    )

    return qa_chain