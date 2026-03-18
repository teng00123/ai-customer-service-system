"""基于内容的推荐算法 - Claude Code风格"""
import logging
import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import re

from .models import UserProfile, ItemProfile, RecommendationResult

logger = logging.getLogger(__name__)

class ContentBasedRecommender:
    """基于内容的推荐算法"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.tfidf_max_features = config.get('tfidf_max_features', 5000)
        self.similarity_threshold = config.get('similarity_threshold', 0.2)
        self.title_weight = config.get('title_weight', 0.3)
        self.content_weight = config.get('content_weight', 0.7)
        
        # 文本向量化器
        self.vectorizer = TfidfVectorizer(
            max_features=self.tfidf_max_features,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2
        )
        
        # 物品内容索引
        self.item_contents = {}  # item_id -> combined_text
        self.item_vectors = {}   # item_id -> vector
        self.category_items = defaultdict(list)  # category -> [item_ids]
        
        # 统计信息
        self.stats = {
            'recommendations_generated': 0,
            'similarity_computations': 0,
            'categories_indexed': 0
        }
        
        logger.info("Content-based recommender initialized")
    
    async def recommend(self, user_profile: UserProfile, top_k: int, 
                       context: Dict[str, Any] = None) -> RecommendationResult:
        """基于内容生成推荐"""
        try:
            user_id = user_profile.user_id
            
            # 分析用户偏好
            preferred_categories = self._analyze_user_preferences(user_profile)
            
            if not preferred_categories:
                # 用户偏好不明确，推荐热门物品
                return await self._recommend_popular_items(user_id, top_k, context)
            
            # 为每个偏好类别生成候选推荐
            candidate_items = []
            for category, weight in preferred_categories.items():
                category_items = await self._get_category_items(category, top_k * 2)
                for item_id in category_items:
                    candidate_items.append((item_id, weight))
            
            # 去重并按权重排序
            seen_items = set()
            unique_candidates = []
            for item_id, weight in candidate_items:
                if item_id not in seen_items:
                    unique_candidates.append((item_id, weight))
                    seen_items.add(item_id)
            
            unique_candidates.sort(key=lambda x: x[1], reverse=True)
            
            # 取前K个物品
            selected_items = [item_id for item_id, _ in unique_candidates[:top_k]]
            selected_scores = [weight * 0.8 for _, weight in unique_candidates[:top_k]]  # 归一化分数
            
            self.stats['recommendations_generated'] += 1
            
            return RecommendationResult(
                user_id=user_id,
                items=selected_items,
                scores=selected_scores,
                algorithm_used="content_based",
                context=context,
                requested_count=top_k
            )
            
        except Exception as e:
            logger.error(f"Content-based recommendation failed: {e}")
            return RecommendationResult.empty(user_profile.user_id)
    
    async def find_similar_items(self, item_profile: ItemProfile, top_k: int, 
                               context: Dict[str, Any] = None) -> RecommendationResult:
        """查找相似物品"""
        try:
            item_id = item_profile.item_id
            
            if item_id not in self.item_vectors:
                # 物品不在索引中，返回热门物品
                return await self._recommend_popular_items("", top_k, context)
            
            # 计算与所有其他物品的相似度
            similarities = []
            target_vector = self.item_vectors[item_id]
            
            for other_id, other_vector in self.item_vectors.items():
                if other_id == item_id:
                    continue
                    
                similarity = cosine_similarity([target_vector], [other_vector])[0][0]
                if similarity >= self.similarity_threshold:
                    similarities.append((other_id, similarity))
            
            # 排序并取前K个
            similarities.sort(key=lambda x: x[1], reverse=True)
            top_items = similarities[:top_k]
            
            items = [item_id for item_id, _ in top_items]
            scores = [score for _, score in top_items]
            
            return RecommendationResult(
                user_id="",  # 基于物品的推荐没有特定用户
                items=items,
                scores=scores,
                algorithm_used="content_based_similarity",
                context=context,
                requested_count=top_k
            )
            
        except Exception as e:
            logger.error(f"Similar item search failed: {e}")
            return RecommendationResult.empty()
    
    def _analyze_user_preferences(self, user_profile: UserProfile) -> Dict[str, float]:
        """分析用户偏好"""
        preferences = {}
        
        # 基于显式偏好
        for category, weight in user_profile.preferences.items():
            if weight > 0.3:  # 只考虑较强的偏好
                preferences[category] = weight
        
        # 基于交互历史推断偏好
        recent_interactions = user_profile.get_recent_interactions(30)
        for interaction in recent_interactions:
            category = interaction.get('feedback', {}).get('category', 'general')
            rating = interaction.get('feedback', {}).get('rating', 3.0)
            
            # 将评分转换为权重 (1-5 映射到 0-1)
            weight = (rating - 1) / 4.0
            
            if category in preferences:
                preferences[category] = 0.7 * preferences[category] + 0.3 * weight
            else:
                preferences[category] = weight
        
        return preferences
    
    async def _get_category_items(self, category: str, limit: int) -> List[str]:
        """获取指定类别的物品"""
        if category in self.category_items:
            return self.category_items[category][:limit]
        return []
    
    async def _recommend_popular_items(self, user_id: str, top_k: int, 
                                      context: Dict[str, Any] = None) -> RecommendationResult:
        """推荐热门物品"""
        # 简化实现 - 返回模拟热门物品
        popular_items = [f"popular_item_{i}" for i in range(top_k)]
        popular_scores = [0.9 - (i * 0.05) for i in range(top_k)]
        
        return RecommendationResult(
            user_id=user_id,
            items=popular_items,
            scores=popular_scores,
            algorithm_used="content_based_popular",
            context=context,
            requested_count=top_k
        )
    
    def index_item(self, item_profile: ItemProfile) -> None:
        """索引物品内容"""
        try:
            # 组合文本内容
            title = item_profile.features.get('title', '')
            description = item_profile.features.get('description', '')
            category = item_profile.category
            
            combined_text = f"{title} {description} {category}".strip()
            self.item_contents[item_profile.item_id] = combined_text
            self.category_items[category].append(item_profile.item_id)
            
            # 更新统计
            self.stats['categories_indexed'] = len(self.category_items)
            
        except Exception as e:
            logger.error(f"Failed to index item {item_profile.item_id}: {e}")
    
    def build_index(self) -> None:
        """构建TF-IDF索引"""
        try:
            if not self.item_contents:
                logger.warning("No items to index")
                return
                
            # 准备文档集合
            documents = []
            item_ids = []
            
            for item_id, content in self.item_contents.items():
                if content.strip():  # 跳过空内容
                    documents.append(content)
                    item_ids.append(item_id)
            
            if not documents:
                logger.warning("No valid documents to index")
                return
            
            # 构建TF-IDF矩阵
            tfidf_matrix = self.vectorizer.fit_transform(documents)
            
            # 存储物品向量
            for i, item_id in enumerate(item_ids):
                vector = tfidf_matrix[i].toarray()[0]
                self.item_vectors[item_id] = vector
            
            logger.info(f"Built TF-IDF index for {len(documents)} items")
            
        except Exception as e:
            logger.error(f"Failed to build index: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            **self.stats,
            'indexed_items': len(self.item_contents),
            'indexed_categories': len(self.category_items),
            'vector_dimension': self.tfidf_max_features
        }