lightweight_ollama_models = [
    {
        "name": "tinyllama",
        "params": "1.1B",
        "pull_command": "ollama pull tinyllama",
        "context_window": "2048 tokens",
        "use_case": "Basic Q&A, prototyping",
        "strengths": "Fastest inference, runs on Raspberry Pi",
        "ram_required": "2GB+",
        "license": "Apache-2.0",
        "local_availability" : True
    },
    {
        "name": "phi3",
        "params": "3.8B",
        "pull_command": "ollama pull phi3",
        "context_window": "4K tokens",
        "use_case": "Structured responses, reasoning tasks",
        "strengths": "Microsoft-optimized, good instruction following",
        "ram_required": "4GB+",
        "license": "MIT",
        "local_availability" : True
    },
    {
        "name": "mistral",
        "params": "7B",
        "pull_command": "ollama pull mistral",
        "context_window": "8K tokens",
        "use_case": "General chatbot applications",
        "strengths": "Best performance/size ratio",
        "ram_required": "6GB+",
        "license": "Apache-2.0",
        "local_availability" : True
    },
    {
        "name": "gemma:2b",
        "params": "2B",
        "pull_command": "ollama pull gemma:2b",
        "context_window": "8K tokens",
        "use_case": "Mobile/edge deployment",
        "strengths": "Google's lightweight model, good safety features",
        "ram_required": "3GB+",
        "license": "Gemma Terms",
        "local_availability" : True
    },
    {
        "name": "stablelm-zephyr",
        "params": "3B",
        "pull_command": "ollama pull stablelm-zephyr",
        "context_window": "4K tokens",
        "use_case": "Conversational AI",
        "strengths": "Fine-tuned for dialogues, permissive license",
        "ram_required": "4GB+",
        "license": "Apache-2.0",
        "local_availability" : True
    },
    {
        "name": "llama3:8b-instruct",
        "params": "7B",
        "pull_command": "ollama pull llama3:8b-instruct",
        "context_window": "8K tokens",
        "use_case": "Instruction following",
        "strengths": "Meta's latest, good general-purpose performance",
        "ram_required": "6GB+",
        "license": "Meta Llama 3 License",
        "local_availability" : False
    },
    {
        "name": "openchat",
        "params": "3.5B",
        "pull_command": "ollama pull openchat",
        "context_window": "4K tokens",
        "use_case": "Multi-turn conversations",
        "strengths": "Optimized for chat history",
        "ram_required": "4GB+",
        "license": "Apache-2.0",
        "local_availability" : False
    },
    {
        "name": "qwen:1.8b",
        "params": "1.8B",
        "pull_command": "ollama pull qwen:1.8b",
        "context_window": "2K tokens",
        "use_case": "Bilingual (Chinese/English) applications",
        "strengths": "Alibaba's compact multilingual model",
        "ram_required": "3GB+",
        "license": "Qwen License",
        "local_availability" : False
    }
]

embeddings_models = {
    # Lightweight (Good for CPU/GPU)
    "all-minilm-l6-v2": {"type": "sentence-transformers", "model": "all-MiniLM-L6-v2"},
    "bge-small-en": {"type": "sentence-transformers", "model": "BAAI/bge-small-en-v1.5"},
    
    # GPU-Optimized
    "bge-base-en": {"type": "sentence-transformers", "model": "BAAI/bge-base-en-v1.5"},
    "paraphrase-multilingual-mpnet": {"type": "sentence-transformers", "model": "paraphrase-multilingual-mpnet-base-v2"},
    
    # Ollama (GPU via Ollama)
    "nomic-embed-text": {"type": "ollama", "model": "nomic-embed-text"},
    "mxbai-embed-large": {"type": "ollama", "model": "mxbai-embed-large"},
}

vectorDB_Path = "vectors/"

chunk_sizes = [256, 512, 800]
chunk_overlaps = [50, 75, 100]
vector_store_type = ["faiss", "chroma", "lancedb"]
benchmark_questions = [
    "Why are there different prices for the same product on Myntra? Is that even legal?",
    "I saw a product listed for ₹1000 but when I clicked on it, the size I want is ₹1600. Why the price change?",
    "Why doesn’t Myntra always show the lowest price on the product listing page?",
    "How can I identify if an email or call claiming to be from Myntra is a scam?",
    "Someone called saying I won a lucky draw from Myntra and asked for my card details. Is this real?",
    "I received a job offer from Myntra but they’re asking for a processing fee. Is this genuine?",
    "How do I know if an appointment letter from Myntra is real?",
    "Why can’t I see ‘My Cashback’ in my Myntra account anymore?",
    "What is PhonePe’s role in Myntra cashback now? Do I need to activate something?",
    "How can I cancel an order I just placed on Myntra?"
]