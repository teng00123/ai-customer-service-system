"""协同过滤算法 - Claude Code风格"""
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from .models import UserProfile, ItemProfile, RecommendationResult

logger = logging.getLogger(__name__)

class CollaborativeFiltering:
    """协同过滤推荐算法"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.similarity_threshold = config.get('similarity_threshold', 0.3)
        self.max_neighbors = config.get('max_neighbors', 50)
        self.min_interactions = config.get('min_interactions', 3)
        
        # 用户-物品交互矩阵
        self.user_item_matrix = defaultdict(dict)  # user_id -> {item_id: rating}
        self.item_user_matrix = defaultdict(dict)  # item_id -> {user_id: rating}
        
        # 用户相似度矩阵
        self.user_similarity_matrix = defaultdict(dict)  # user_id -> {other_user_id: similarity}
        
        # 统计信息
        self.stats = {
            'recommendations_generated': 0,
            'similarity_computations': 0,
            'cache_hits': 0
        }
        
        logger.info("Collaborative filtering initialized")
    
    async def recommend(self, user_profile: UserProfile, top_k: int, 
                       context: Dict[str, Any] = None) -> RecommendationResult:
        """基于协同过滤生成推荐"""
        try:
            user_id = user_profile.user_id
            
            # 获取目标用户的评分历史
            user_ratings = self.user_item_matrix.get(user_id, {})
            
            if len(user_ratings) < self.min_interactions:
                # 用户交互太少，回退到热门推荐
                return await self._fallback_to_popular(user_id, top_k, context)
            
            # 找到相似用户
            similar_users = await self._find_similar_users(user_id, user_ratings)
            
            if not similar_users:
                return await self._fallback_to_popular(user_id, top_k, context)
            
            # 基于相似用户的评分预测目标用户可能喜欢的物品
            predictions = await self._predict_ratings(user_id, similar_users, top_k)
            
            # 排序并取前K个
            sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:top_k]
            
            items = [item_id for item_id, _ in sorted_predictions]
            scores = [score for _, score in sorted_predictions]
            
            self.stats['recommendations_generated'] += 1
            
            return RecommendationResult(
                user_id=user_id,
                items=items,
                scores=scores,
                algorithm_used="collaborative_filtering",
                context=context,
                requested_count=top_k
            )
            
        except Exception as e:
            logger.error(f"Collaborative filtering recommendation failed: {e}")
            return RecommendationResult.empty(user_profile.user_id)
    
    async def _find_similar_users(self, target_user_id: str, target_ratings: Dict[str, float]) -> List[Tuple[str, float]]:
        """找到与目标用户相似的用户"""
        similar_users = []
        
        # 并行计算相似度
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for other_user_id, other_ratings in self.user_item_matrix.items():
                if other_user_id != target_user_id and len(other_ratings) >= self.min_interactions:
                    future = executor.submit(
                        self._calculate_user_similarity,
                        target_ratings, other_ratings
                    )
                    futures.append((other_user_id, future))
            
            for other_user_id, future in futures:
                try:
                    similarity = future.result(timeout=5.0)
                    if similarity >= self.similarity_threshold:
                        similar_users.append((other_user_id, similarity))
                except Exception as e:
                    logger.warning(f"Similarity calculation failed for {other_user_id}: {e}")
        
        # 按相似度排序，取前N个
        similar_users.sort(key=lambda x: x[1], reverse=True)
        return similar_users[:self.max_neighbors]
    
    def _calculate_user_similarity(self, ratings1: Dict[str, float], ratings2: Dict[str, float]) -> float:
        """计算两个用户的相似度 (皮尔逊相关系数)"""
        self.stats['similarity_computations'] += 1
        
        # 找到共同评分的物品
        common_items = set(ratings1.keys()) & set(ratings2.keys())
        
        if len(common_items) < 2:
            return 0.0
        
        # 提取共同物品的评分
        r1 = [ratings1[item] for item in common_items]
        r2 = [ratings2[item] for item in common_items]
        
        # 计算皮尔逊相关系数
        mean1 = np.mean(r1)
        mean2 = np.mean(r2)
        
        numerator = sum((x - mean1) * (y - mean2) for x, y in zip(r1, r2))
        denominator1 = np.sqrt(sum((x - mean1) ** 2 for x in r1))
        denominator2 = np.sqrt(sum((y - mean2) ** 2 for y in r2))
        
        if denominator1 == 0 or denominator2 == 0:
            return 0.0
        
        similarity = numerator / (denominator1 * denominator2)
        return max(-1.0, min(1.0, similarity))  # 限制在[-1, 1]范围内
    
    async def _predict_ratings(self, target_user_id: str, similar_users: List[Tuple[str, float]], 
                              top_k: int) -> Dict[str, float]:
        """预测目标用户对未评分物品的评分"""
        predictions = defaultdict(float)
        prediction_counts = defaultdict(int)
        
        target_ratings = self.user_item_matrix.get(target_user_id, {})
        
        for similar_user_id, similarity in similar_users:
            similar_ratings = self.user_item_matrix.get(similar_user_id, {})
            
            for item_id, rating in similar_ratings.items():
                # 跳过目标用户已经评分的物品
                if item_id in target_ratings:
                    continue
                
                # 加权评分预测
                weighted_rating = similarity * rating
                predictions[item_id] += weighted_rating
                prediction_counts[item_id] += abs(similarity)
        
        # 计算加权平均评分
        final_predictions = {}
        for item_id in predictions:
            if prediction_counts[item_id] > 0:
                final_predictions[item_id] = predictions[item_id] / prediction_counts[item_id]
        
        return final_predictions
    
    async def _fallback_to_popular(self, user_id: str, top_k: int, 
                                  context: Dict[str, Any] = None) -> RecommendationResult:
        """回退到热门推荐"""
        # 获取热门物品 (简化实现)
        popular_items = [f"popular_item_{i}" for i in range(top_k)]
        popular_scores = [0.8 - (i * 0.05) for i in range(len(popular_items))]
        
        return RecommendationResult(
            user_id=user_id,
            items=popular_items,
            scores=popular_scores,
            algorithm_used="collaborative_filtering_fallback",
            context=context,
            requested_count=top_k
        )
    
    def update_user_item_interaction(self, user_id: str, item_id: str, rating: float) -> None:
        """更新用户-物品交互"""
        self.user_item_matrix[user_id][item_id] = rating
        self.item_user_matrix[item_id][user_id] = rating
        
        # 清除相似度缓存
        if user_id in self.user_similarity_matrix:
            del self.user_similarity_matrix[user_id]
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            **self.stats,
            'user_item_pairs': len(self.user_item_matrix),
            'total_interactions': sum(len(ratings) for ratings in self.user_item_matrix.values()),
            'unique_items': len(self.item_user_matrix)
        }