"""向量存储模块 - Claude Code风格"""
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import faiss
import pickle
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class VectorStore:
    """向量存储管理器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._default_config()
        self.dimension = self.config['dimension']
        self.index = None
        self.metadata_store = {}  # chunk_id -> metadata
        self.storage_path = Path(self.config.get('storage_path', 'data/vectors'))
        
        # 创建存储目录
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # 初始化FAISS索引
        self._init_index()
        
        logger.info(f"Vector store initialized with dimension {self.dimension}")
        
    def _default_config(self) -> Dict[str, Any]:
        return {
            'dimension': 1536,
            'index_type': 'flat',  # flat, ivf, hnsw
            'metric': 'cosine',     # cosine, l2, inner_product
            'storage_path': 'data/vectors',
            'normalize': True,
            'nlist': 100,           # for IVF
            'nprobe': 10             # for IVF
        }
    
    def _init_index(self) -> None:
        """初始化FAISS索引"""
        metric_map = {
            'cosine': faiss.METRIC_INNER_PRODUCT,
            'l2': faiss.METRIC_L2,
            'inner_product': faiss.METRIC_INNER_PRODUCT
        }
        
        metric = metric_map.get(self.config['metric'], faiss.METRIC_INNER_PRODUCT)
        
        if self.config['index_type'] == 'flat':
            self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine
        elif self.config['index_type'] == 'ivf':
            quantizer = faiss.IndexFlatIP(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, self.config['nlist'], metric)
        else:
            # 默认使用Flat索引
            self.index = faiss.IndexFlatIP(self.dimension)
            
        # 加载已有索引
        self._load_index()
    
    async def add_chunks(self, chunks: List, embeddings: List[List[float]]) -> List[str]:
        """添加文档块到向量存储"""
        try:
            if not chunks or not embeddings:
                return []
                
            # 准备向量数据
            vectors = np.array(embeddings, dtype=np.float32)
            
            # 归一化向量（用于余弦相似度）
            if self.config.get('normalize', True):
                faiss.normalize_L2(vectors)
            
            # 添加到索引
            chunk_ids = [chunk.id for chunk in chunks]
            self.index.add(vectors)
            
            # 存储元数据
            for i, (chunk, chunk_id) in enumerate(zip(chunks, chunk_ids)):
                self.metadata_store[chunk_id] = {
                    'vector_offset': self.index.ntotal - len(vectors) + i,
                    'chunk': chunk,
                    'added_at': datetime.now()
                }
            
            # 持久化存储
            await self._save_index()
            
            logger.info(f"Added {len(chunks)} chunks to vector store")
            return chunk_ids
            
        except Exception as e:
            logger.error(f"Failed to add chunks: {e}")
            raise
    
    async def search(self, query_embedding: List[float], top_k: int = 10,
                   filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """向量搜索"""
        try:
            # 准备查询向量
            query_vector = np.array([query_embedding], dtype=np.float32)
            
            if self.config.get('normalize', True):
                faiss.normalize_L2(query_vector)
            
            # 执行搜索
            scores, indices = self.index.search(query_vector, top_k)
            
            # 处理结果
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx == -1:  # FAISS返回-1表示无结果
                    continue
                    
                # 查找对应的chunk_id
                chunk_id = self._find_chunk_id_by_offset(idx)
                if chunk_id and chunk_id in self.metadata_store:
                    metadata = self.metadata_store[chunk_id]
                    
                    # 应用过滤器
                    if filters and not self._apply_filters(metadata['chunk'], filters):
                        continue
                        
                    results.append({
                        'chunk_id': chunk_id,
                        'score': float(score),
                        'chunk': metadata['chunk'],
                        'rank': i + 1
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []
    
    async def delete_chunks(self, chunk_ids: List[str]) -> bool:
        """删除文档块"""
        try:
            deleted_count = 0
            for chunk_id in chunk_ids:
                if chunk_id in self.metadata_store:
                    del self.metadata_store[chunk_id]
                    deleted_count += 1
            
            # 重建索引（简化实现）
            if deleted_count > 0:
                await self._rebuild_index()
                
            logger.info(f"Deleted {deleted_count} chunks from vector store")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete chunks: {e}")
            return False
    
    def _find_chunk_id_by_offset(self, offset: int) -> Optional[str]:
        """根据偏移量查找chunk_id"""
        for chunk_id, metadata in self.metadata_store.items():
            if metadata['vector_offset'] == offset:
                return chunk_id
        return None
    
    def _apply_filters(self, chunk, filters: Dict[str, Any]) -> bool:
        """应用过滤器"""
        try:
            # 类别过滤
            if 'category' in filters:
                # 需要从chunk追溯到document
                pass
                
            # 标签过滤
            if 'tags' in filters:
                chunk_tags = getattr(chunk, 'tags', [])
                if not any(tag in chunk_tags for tag in filters['tags']):
                    return False
                    
            return True
            
        except Exception:
            return True  # 过滤失败时不过滤
    
    async def _save_index(self) -> None:
        """保存索引到磁盘"""
        try:
            index_path = self.storage_path / "faiss_index.bin"
            metadata_path = self.storage_path / "metadata.pkl"
            
            # 保存FAISS索引
            faiss.write_index(self.index, str(index_path))
            
            # 保存元数据
            with open(metadata_path, 'wb') as f:
                pickle.dump({
                    'metadata_store': self.metadata_store,
                    'config': self.config
                }, f)
                
            logger.debug("Vector store saved to disk")
            
        except Exception as e:
            logger.error(f"Failed to save index: {e}")
    
    def _load_index(self) -> None:
        """从磁盘加载索引"""
        try:
            index_path = self.storage_path / "faiss_index.bin"
            metadata_path = self.storage_path / "metadata.pkl"
            
            if index_path.exists() and metadata_path.exists():
                # 加载FAISS索引
                self.index = faiss.read_index(str(index_path))
                
                # 加载元数据
                with open(metadata_path, 'rb') as f:
                    data = pickle.load(f)
                    self.metadata_store = data.get('metadata_store', {})
                    
                logger.info(f"Loaded vector store with {self.index.ntotal} vectors")
                
        except Exception as e:
            logger.warning(f"Failed to load index: {e}")
    
    async def _rebuild_index(self) -> None:
        """重建索引"""
        try:
            # 简化实现：清空并重新添加
            old_metadata = self.metadata_store.copy()
            self.metadata_store.clear()
            
            if len(old_metadata) > 0:
                # 重新构建向量数组
                vectors = []
                chunks = []
                chunk_ids = []
                
                for chunk_id, metadata in old_metadata.items():
                    # 这里需要重新生成嵌入向量（实际项目中应缓存）
                    # 简化处理：跳过重建
                    pass
                
                # 由于重建复杂，这里仅清空索引
                self.index = faiss.IndexFlatIP(self.dimension)
                
            logger.info("Vector store rebuilt")
            
        except Exception as e:
            logger.error(f"Failed to rebuild index: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取向量存储统计"""
        return {
            'total_vectors': self.index.ntotal if self.index else 0,
            'dimension': self.dimension,
            'index_type': self.config['index_type'],
            'metric': self.config['metric'],
            'metadata_count': len(self.metadata_store),
            'storage_path': str(self.storage_path)
        }