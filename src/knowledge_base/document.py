"""知识库文档模型 - Claude Code风格"""
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

class DocumentStatus(Enum):
    """文档状态枚举"""
    DRAFT = "draft"
    PUBLISHED = "published"
    ARCHIVED = "archived"
    DELETED = "deleted"

@dataclass
class DocumentChunk:
    """文档块模型"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    document_id: str = ""
    chunk_index: int = 0
    embedding_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'id': self.id,
            'content': self.content,
            'document_id': self.document_id,
            'chunk_index': self.chunk_index,
            'embedding_id': self.embedding_id,
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat()
        }

@dataclass
class Document:
    """文档模型"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    content: str = ""
    category: str = "general"
    tags: List[str] = field(default_factory=list)
    status: DocumentStatus = DocumentStatus.PUBLISHED
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    author: str = ""
    version: str = "1.0.0"
    language: str = "zh-CN"
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'id': self.id,
            'title': self.title,
            'content': self.content,
            'category': self.category,
            'tags': self.tags,
            'status': self.status.value,
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'author': self.author,
            'version': self.version,
            'language': self.language
        }
    
    def get_summary(self, max_length: int = 200) -> str:
        """获取文档摘要"""
        if len(self.content) <= max_length:
            return self.content
        return self.content[:max_length] + "..."
    
    def add_tag(self, tag: str) -> None:
        """添加标签"""
        if tag not in self.tags:
            self.tags.append(tag)
            self.updated_at = datetime.now()
    
    def remove_tag(self, tag: str) -> None:
        """移除标签"""
        if tag in self.tags:
            self.tags.remove(tag)
            self.updated_at = datetime.now()