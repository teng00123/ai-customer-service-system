"""知识库管理器 - Claude Code风格"""
import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from .document import Document, DocumentChunk
from .search import SearchEngine
from .vector_store import VectorStore
from ..core.config import settings

logger = logging.getLogger(__name__)

class KnowledgeBaseManager:
    """知识库管理器主控制器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._default_config()
        self.vector_store = VectorStore(self.config.get('vector_store', {}))
        self.search_engine = SearchEngine(self.config.get('search', {}))
        self.documents = {}  # 文档缓存
        self.document_chunks = {}  # 文档块缓存
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # 统计信息
        self.stats = {
            'total_documents': 0,
            'total_chunks': 0,
            'total_searches': 0,
            'average_search_time': 0.0,
            'last_updated': datetime.now()
        }
        
        logger.info("Knowledge base manager initialized")
        
    def _default_config(self) -> Dict[str, Any]:
        """默认配置"""
        return {
            'vector_store': {
                'embedding_model': 'text-embedding-ada-002',
                'dimension': 1536,
                'index_type': 'faiss',
                'metric': 'cosine'
            },
            'search': {
                'max_results': 10,
                'min_score': 0.7,
                'enable_reranking': True,
                'rerank_top_k': 20
            },
            'chunking': {
                'chunk_size': 500,
                'chunk_overlap': 50,
                'separator': '\n\n'
            }
        }
    
    async def add_document(self, title: str, content: str, category: str = "general",
                         tags: List[str] = None, metadata: Dict[str, Any] = None) -> str:
        """添加文档到知识库"""
        try:
            # 创建文档对象
            doc = Document(
                title=title,
                content=content,
                category=category,
                tags=tags or [],
                metadata=metadata or {}
            )
            
            # 文档分块
            chunks = await self._chunk_document(doc)
            
            # 生成嵌入向量
            embeddings = await self._generate_embeddings([chunk.content for chunk in chunks])
            
            # 存储到向量数据库
            chunk_ids = await self.vector_store.add_chunks(chunks, embeddings)
            
            # 更新缓存
            self.documents[doc.id] = doc
            for i, chunk in enumerate(chunks):
                chunk.embedding_id = chunk_ids[i]
                self.document_chunks[chunk.id] = chunk
            
            # 更新统计
            self.stats['total_documents'] += 1
            self.stats['total_chunks'] += len(chunks)
            
            logger.info(f"Added document: {doc.id} ({title}) with {len(chunks)} chunks")
            return doc.id
            
        except Exception as e:
            logger.error(f"Failed to add document: {e}")
            raise
    
    async def _chunk_document(self, document: Document) -> List[DocumentChunk]:
        """文档分块处理"""
        chunk_size = self.config['chunking']['chunk_size']
        overlap = self.config['chunking']['chunk_overlap']
        separator = self.config['chunking']['separator']
        
        # 按分隔符分割
        sections = document.content.split(separator)
        chunks = []
        current_chunk = ""
        chunk_index = 0
        
        for section in sections:
            if len(current_chunk) + len(section) <= chunk_size:
                current_chunk += section + separator
            else:
                if current_chunk.strip():
                    chunk = DocumentChunk(
                        content=current_chunk.strip(),
                        document_id=document.id,
                        chunk_index=chunk_index,
                        metadata={"section": chunk_index}
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                
                # 处理剩余内容
                current_chunk = section + separator
        
        # 添加最后一个块
        if current_chunk.strip():
            chunk = DocumentChunk(
                content=current_chunk.strip(),
                document_id=document.id,
                chunk_index=chunk_index,
                metadata={"section": chunk_index}
            )
            chunks.append(chunk)
        
        return chunks
    
    async def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """生成文本嵌入向量"""
        # 模拟嵌入生成（实际项目中应调用真实的嵌入模型）
        embeddings = []
        for text in texts:
            # 简化的随机嵌入（实际应使用OpenAI/Azure等API）
            embedding = np.random.rand(self.config['vector_store']['dimension']).tolist()
            # 归一化
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = (np.array(embedding) / norm).tolist()
            embeddings.append(embedding)
            
        return embeddings
    
    async def search(self, query: str, top_k: int = 5, 
                   filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """语义搜索"""
        start_time = datetime.now()
        
        try:
            # 生成查询嵌入
            query_embeddings = await self._generate_embeddings([query])
            query_embedding = query_embeddings[0]
            
            # 向量搜索
            vector_results = await self.vector_store.search(
                query_embedding, top_k * 2, filters
            )
            
            # 重排序
            if self.config['search']['enable_reranking']:
                final_results = await self.search_engine.rerank(
                    query, vector_results, top_k
                )
            else:
                final_results = vector_results[:top_k]
            
            # 更新统计
            search_time = (datetime.now() - start_time).total_seconds()
            self._update_search_stats(search_time)
            
            logger.info(f"Search completed: query='{query[:50]}...', results={len(final_results)}")
            return final_results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    async def get_document(self, doc_id: str) -> Optional[Document]:
        """获取文档"""
        return self.documents.get(doc_id)
    
    async def update_document(self, doc_id: str, updates: Dict[str, Any]) -> bool:
        """更新文档"""
        try:
            if doc_id not in self.documents:
                return False
                
            doc = self.documents[doc_id]
            
            # 更新字段
            for key, value in updates.items():
                if hasattr(doc, key):
                    setattr(doc, key, value)
            
            doc.updated_at = datetime.now()
            
            # 重新索引（如果需要）
            if 'content' in updates:
                await self._reindex_document(doc)
                
            logger.info(f"Updated document: {doc_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update document {doc_id}: {e}")
            return False
    
    async def delete_document(self, doc_id: str) -> bool:
        """删除文档"""
        try:
            if doc_id not in self.documents:
                return False
                
            # 删除相关块
            chunk_ids = [chunk.id for chunk in self.document_chunks.values() if chunk.document_id == doc_id]
            await self.vector_store.delete_chunks(chunk_ids)
            
            # 删除缓存
            del self.documents[doc_id]
            for chunk_id in chunk_ids:
                if chunk_id in self.document_chunks:
                    del self.document_chunks[chunk_id]
            
            # 更新统计
            self.stats['total_documents'] -= 1
            self.stats['total_chunks'] -= len(chunk_ids)
            
            logger.info(f"Deleted document: {doc_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete document {doc_id}: {e}")
            return False
    
    async def _reindex_document(self, document: Document) -> None:
        """重新索引文档"""
        # 删除旧块
        old_chunks = [chunk for chunk in self.document_chunks.values() if chunk.document_id == document.id]
        old_chunk_ids = [chunk.id for chunk in old_chunks]
        await self.vector_store.delete_chunks(old_chunk_ids)
        
        # 添加新块
        new_chunks = await self._chunk_document(document)
        embeddings = await self._generate_embeddings([chunk.content for chunk in new_chunks])
        chunk_ids = await self.vector_store.add_chunks(new_chunks, embeddings)
        
        # 更新缓存
        for i, chunk in enumerate(new_chunks):
            chunk.embedding_id = chunk_ids[i]
            self.document_chunks[chunk.id] = chunk
    
    def _update_search_stats(self, search_time: float) -> None:
        """更新搜索统计"""
        self.stats['total_searches'] += 1
        total_searches = self.stats['total_searches']
        current_avg = self.stats['average_search_time']
        self.stats['average_search_time'] = (
            (current_avg * (total_searches - 1) + search_time) / total_searches
        )
        self.stats['last_updated'] = datetime.now()
    
    def get_stats(self) -> Dict[str, Any]:
        """获取知识库统计"""
        return {
            **self.stats,
            'vector_store_stats': self.vector_store.get_stats(),
            'search_engine_stats': self.search_engine.get_stats()
        }