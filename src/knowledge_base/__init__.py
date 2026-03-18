"""知识库模块 - Claude Code风格实现"""
from .manager import KnowledgeBaseManager
from .document import Document, DocumentChunk
from .search import SearchEngine, SearchResult
from .vector_store import VectorStore

__all__ = [
    "KnowledgeBaseManager",
    "Document", 
    "DocumentChunk",
    "SearchEngine",
    "SearchResult",
    "VectorStore"
]