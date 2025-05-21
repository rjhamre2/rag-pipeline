from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import List, Dict, Any
import hashlib
import re

class DocumentChunker:
    def __init__(
        self,
        chunk_size: int = 1024,
        chunk_overlap: int = 128,
        separators: List[str] = None
    ):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators or ["\n\n", "\n", "(?<=\. )", " ", ""],
            length_function=len,
            add_start_index=True,
            is_separator_regex=True
        )

    def chunk_documents(
        self,
        documents: List[Document],
        extra_metadata: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """Split documents with enhanced metadata tracking"""
        chunks = self.splitter.split_documents(documents)
        
        processed_chunks = []
        for doc_idx, chunk in enumerate(chunks):
            chunk_id = hashlib.md5(
                f"{doc_idx}-{chunk.metadata.get('start_index', 0)}".encode()
            ).hexdigest()
            
            processed_chunks.append({
                "id": chunk_id,
                "text": self._clean_text(chunk.page_content),
                "metadata": {
                    **(extra_metadata or {}),
                    **chunk.metadata,
                    "doc_idx": doc_idx,
                    "start_char": chunk.metadata.get("start_index", 0),
                    "end_char": chunk.metadata.get("start_index", 0) + len(chunk.page_content),
                    "is_continuation": chunk.metadata.get("start_index", 0) > 0
                }
            })
        return processed_chunks

    def _clean_text(self, text: str) -> str:
        """Basic text normalization"""
        text = re.sub(r'\s+', ' ', text).strip()
        return text