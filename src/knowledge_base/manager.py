import logging
from typing import Dict, List, Any, Optional
from elasticsearch import Elasticsearch
from ..core.config import settings

logger = logging.getLogger(__name__)

class KnowledgeBaseManager:
    """知识库管理系统"""
    
    def __init__(self, es_hosts: str = None):
        self.es_hosts = es_hosts or settings.elasticsearch_hosts
        self.es = Elasticsearch([self.es_hosts])
        self.index_name = "knowledge_base"
        
    def create_index(self) -> bool:
        """创建知识库索引"""
        index_mapping = {
            "mappings": {
                "properties": {
                    "question": {"type": "text", "analyzer": "ik_max_word"},
                    "answer": {"type": "text", "analyzer": "ik_max_word"},
                    "category": {"type": "keyword"},
                    "tags": {"type": "keyword"},
                    "created_at": {"type": "date"},
                    "updated_at": {"type": "date"},
                    "views": {"type": "integer"},
                    "helpful_votes": {"type": "integer"}
                }
            }
        }
        
        try:
            if not self.es.indices.exists(index=self.index_name):
                self.es.indices.create(index=self.index_name, body=index_mapping)
                logger.info(f"Created index: {self.index_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to create index: {e}")
            return False
    
    def add_document(self, question: str, answer: str, category: str = "general", 
                   tags: List[str] = None) -> bool:
        """添加知识文档"""
        doc = {
            "question": question,
            "answer": answer,
            "category": category,
            "tags": tags or [],
            "created_at": "now",
            "updated_at": "now",
            "views": 0,
            "helpful_votes": 0
        }
        
        try:
            self.es.index(index=self.index_name, body=doc)
            logger.info(f"Added document: {question[:50]}...")
            return True
        except Exception as e:
            logger.error(f"Failed to add document: {e}")
            return False
    
    def search(self, query: str, size: int = 5) -> List[Dict[str, Any]]:
        """搜索知识库"""
        search_body = {
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["question^2", "answer", "tags"],
                    "type": "best_fields"
                }
            },
            "size": size,
            "sort": [{"_score": {"order": "desc"}}]
        }
        
        try:
            response = self.es.search(index=self.index_name, body=search_body)
            results = []
            for hit in response["hits"]["hits"]:
                result = hit["_source"]
                result["_score"] = hit["_score"]
                result["_id"] = hit["_id"]
                results.append(result)
            return results
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []