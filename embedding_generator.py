from model_list import embeddings_models
from typing import Union
from langchain_community.embeddings import OllamaEmbeddings, HuggingFaceEmbeddings

def get_embedding_model(model_name: str) -> Union[OllamaEmbeddings, HuggingFaceEmbeddings]:
    """
    Get the embedding model based on the model name.

    Args:
        model_name (str): Name of the model (e.g., "bge-small-en", "nomic-embed-text").

    Returns:
        Union[OllamaEmbeddings, HuggingFaceEmbeddings]: Initialized embedding model.

    Raises:
        ValueError: If the model name is not found or the type is unsupported.
    """
    # Check if the model exists
    if model_name not in embeddings_models:
        raise ValueError(f"Model '{model_name}' not found. Supported models: {list(embeddings_models.keys())}")

    model_info = embeddings_models[model_name]
    print(f"Loading model: {model_name} | Type: {model_info['type']} | Model: {model_info['model']}")

    # Initialize the appropriate embedding class
    if model_info["type"] == "ollama":
        return OllamaEmbeddings(model=model_info["model"])
    elif model_info["type"] == "sentence-transformers":
        return HuggingFaceEmbeddings(model_name=model_info["model"])  # Note: HuggingFaceEmbeddings uses `model_name`
    else:
        raise ValueError(f"Unsupported model type: {model_info['type']}")