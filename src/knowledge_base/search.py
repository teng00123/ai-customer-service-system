"""搜索引擎模块 - Claude Code风格"""
import logging
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime
from collections import defaultdict

from .document import DocumentChunk

logger = logging.getLogger(__name__)

class SearchResult:
    """搜索结果模型"""
    def __init__(self, chunk: DocumentChunk, score: float, 
                 highlight: str = "", metadata: Dict[str, Any] = None):
        self.chunk = chunk
        self.score = score
        self.highlight = highlight
        self.metadata = metadata or {}
        self.rank = 0
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            'chunk_id': self.chunk.id,
            'document_id': self.chunk.document_id,
            'title': self.metadata.get('title', ''),
            'content': self.chunk.content,
            'score': self.score,
            'highlight': self.highlight,
            'rank': self.rank,
            'metadata': {**self.chunk.metadata, **self.metadata}
        }

class SearchEngine:
    """搜索引擎核心"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._default_config()
        self.search_history = []  # 搜索历史
        self.popular_queries = defaultdict(int)  # 热门查询
        
        logger.info("Search engine initialized")
        
    def _default_config(self) -> Dict[str, Any]:
        return {
            'max_results': 10,
            'min_score': 0.3,
            'enable_reranking': True,
            'rerank_top_k': 20,
            'enable_highlighting': True,
            'enable_fuzzy_match': True,
            'boost_exact_match': 1.2,
            'boost_title': 1.5
        }
    
    async def search(self, query: str, chunks: List[DocumentChunk], 
                   embeddings: List[List[float]], top_k: int = 5) -> List[SearchResult]:
        """执行搜索"""
        try:
            # 计算相似度分数
            results = []
            query_embedding = np.mean(embeddings[:1], axis=0) if embeddings else np.random.rand(768)
            
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                # 余弦相似度
                similarity = self._cosine_similarity(query_embedding, embedding)
                
                # 关键词匹配加分
                keyword_score = self._keyword_match_score(query, chunk.content)
                final_score = similarity * 0.7 + keyword_score * 0.3
                
                if final_score >= self.config['min_score']:
                    result = SearchResult(
                        chunk=chunk,
                        score=final_score,
                        highlight=self._highlight_matches(query, chunk.content) if self.config['enable_highlighting'] else "",
                        metadata={'similarity': similarity, 'keyword_score': keyword_score}
                    )
                    results.append(result)
            
            # 排序
            results.sort(key=lambda x: x.score, reverse=True)
            
            # 重排序
            if self.config['enable_reranking'] and len(results) > 1:
                results = await self.rerank(query, results, top_k)
            else:
                results = results[:top_k]
            
            # 设置排名
            for i, result in enumerate(results):
                result.rank = i + 1
                
            # 记录搜索历史
            self._record_search(query, len(results))
            
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    async def rerank(self, query: str, results: List[SearchResult], 
                   top_k: int) -> List[SearchResult]:
        """重排序搜索结果"""
        try:
            # 简单的重排序策略：考虑查询-文档匹配度
            for result in results:
                # 标题匹配加分
                if query.lower() in result.metadata.get('title', '').lower():
                    result.score *= self.config['boost_title']
                
                # 精确匹配加分
                if query.lower() == result.chunk.content.lower()[:len(query)]:
                    result.score *= self.config['boost_exact_match']
            
            # 重新排序
            results.sort(key=lambda x: x.score, reverse=True)
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return results[:top_k]
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """计算余弦相似度"""
        try:
            vec1_np = np.array(vec1)
            vec2_np = np.array(vec2)
            
            dot_product = np.dot(vec1_np, vec2_np)
            norm1 = np.linalg.norm(vec1_np)
            norm2 = np.linalg.norm(vec2_np)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
                
            return float(dot_product / (norm1 * norm2))
            
        except Exception:
            return 0.0
    
    def _keyword_match_score(self, query: str, content: str) -> float:
        """计算关键词匹配分数"""
        try:
            query_words = set(query.lower().split())
            content_words = set(content.lower().split())
            
            if not query_words:
                return 0.0
                
            intersection = query_words & content_words
            return len(intersection) / len(query_words)
            
        except Exception:
            return 0.0
    
    def _highlight_matches(self, query: str, content: str) -> str:
        """高亮匹配的文本"""
        highlighted = content
        for word in query.split():
            if word.lower() in content.lower():
                highlighted = highlighted.replace(word, f"**{word}**")
        return highlighted
    
    def _record_search(self, query: str, result_count: int) -> None:
        """记录搜索历史"""
        self.search_history.append({
            'query': query,
            'timestamp': datetime.now(),
            'result_count': result_count
        })
        
        self.popular_queries[query] += 1
        
        # 保持历史记录长度
        if len(self.search_history) > 1000:
            self.search_history = self.search_history[-1000:]
    
    def get_stats(self) -> Dict[str, Any]:
        """获取搜索引擎统计"""
        return {
            'total_searches': len(self.search_history),
            'popular_queries': dict(sorted(self.popular_queries.items(), key=lambda x: x[1], reverse=True)[:10]),
            'recent_searches': [h['query'] for h in self.search_history[-10:]]
        }