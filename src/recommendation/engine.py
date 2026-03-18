"""推荐引擎主控制器 - Claude Code风格"""
import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import json

from .models import UserProfile, ItemProfile, RecommendationResult
from .collaborative_filtering import CollaborativeFiltering
from .content_based import ContentBasedRecommender
from .hybrid import HybridRecommender
from ..core.config import settings
from ..utils.cache import CacheManager

logger = logging.getLogger(__name__)

class RecommendationEngine:
    """推荐引擎主控制器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._default_config()
        self.cache = CacheManager(self.config.get('cache', {}))
        
        # 初始化推荐算法
        self.collaborative_filtering = CollaborativeFiltering(
            self.config.get('collaborative_filtering', {})
        )
        self.content_based = ContentBasedRecommender(
            self.config.get('content_based', {})
        )
        self.hybrid = HybridRecommender(
            self.config.get('hybrid', {}),
            self.collaborative_filtering,
            self.content_based
        )
        
        # 用户和物品画像存储
        self.user_profiles = {}  # user_id -> UserProfile
        self.item_profiles = {}  # item_id -> ItemProfile
        
        # 推荐历史记录
        self.recommendation_history = defaultdict(list)
        
        # 性能统计
        self.stats = {
            'total_recommendations': 0,
            'cache_hits': 0,
            'average_response_time': 0.0,
            'algorithm_usage': {
                'collaborative': 0,
                'content_based': 0,
                'hybrid': 0
            },
            'last_updated': datetime.now()
        }
        
        logger.info("Recommendation engine initialized")
        
    def _default_config(self) -> Dict[str, Any]:
        return {
            'cache': {
                'ttl': 3600,  # 1小时缓存
                'max_size': 10000
            },
            'collaborative_filtering': {
                'similarity_threshold': 0.3,
                'max_neighbors': 50,
                'min_interactions': 3
            },
            'content_based': {
                'tfidf_max_features': 5000,
                'similarity_threshold': 0.2,
                'title_weight': 0.3,
                'content_weight': 0.7
            },
            'hybrid': {
                'cf_weight': 0.6,
                'cb_weight': 0.4,
                'diversity_factor': 0.1
            },
            'cold_start': {
                'popular_items_count': 20,
                'random_ratio': 0.1
            }
        }
    
    async def initialize(self) -> None:
        """初始化推荐引擎"""
        try:
            # 加载用户和物品画像
            await self._load_profiles()
            
            # 预热缓存
            await self._warmup_cache()
            
            logger.info("Recommendation engine initialization completed")
            
        except Exception as e:
            logger.error(f"Recommendation engine initialization failed: {e}")
            raise
    
    async def recommend_for_user(self, user_id: str, context: Dict[str, Any] = None,
                                top_k: int = 10, algorithm: str = "hybrid") -> RecommendationResult:
        """为用户生成推荐"""
        start_time = datetime.now()
        
        try:
            context = context or {}
            cache_key = f"rec:{user_id}:{algorithm}:{top_k}:{hash(json.dumps(context, sort_keys=True))}"
            
            # 检查缓存
            cached_result = await self.cache.get(cache_key)
            if cached_result:
                self.stats['cache_hits'] += 1
                logger.debug(f"Cache hit for user {user_id}")
                return RecommendationResult.from_dict(cached_result)
            
            # 获取用户画像
            user_profile = await self._get_user_profile(user_id)
            if not user_profile:
                # 冷启动处理
                result = await self._handle_cold_start(user_id, top_k, context)
            else:
                # 选择推荐算法
                if algorithm == "collaborative":
                    result = await self.collaborative_filtering.recommend(
                        user_profile, top_k, context
                    )
                    self.stats['algorithm_usage']['collaborative'] += 1
                elif algorithm == "content_based":
                    result = await self.content_based.recommend(
                        user_profile, top_k, context
                    )
                    self.stats['algorithm_usage']['content_based'] += 1
                else:  # hybrid (default)
                    result = await self.hybrid.recommend(
                        user_profile, top_k, context
                    )
                    self.stats['algorithm_usage']['hybrid'] += 1
            
            # 后处理
            result = await self._post_process_recommendations(result, user_id, context)
            
            # 缓存结果
            await self.cache.set(cache_key, result.to_dict(), ttl=self.config['cache']['ttl'])
            
            # 记录推荐历史
            self._record_recommendation_history(user_id, result)
            
            # 更新统计
            processing_time = (datetime.now() - start_time).total_seconds()
            self._update_stats(processing_time)
            
            logger.info(f"Generated {len(result.items)} recommendations for user {user_id} using {algorithm}")
            return result
            
        except Exception as e:
            logger.error(f"Recommendation failed for user {user_id}: {e}")
            return RecommendationResult.empty()
    
    async def recommend_for_item(self, item_id: str, top_k: int = 10, 
                               context: Dict[str, Any] = None) -> RecommendationResult:
        """基于物品的协同过滤推荐"""
        try:
            item_profile = await self._get_item_profile(item_id)
            if not item_profile:
                return RecommendationResult.empty()
                
            # 使用基于内容的推荐找到相似物品
            result = await self.content_based.find_similar_items(item_profile, top_k, context)
            
            logger.info(f"Generated {len(result.items)} similar items for {item_id}")
            return result
            
        except Exception as e:
            logger.error(f"Item recommendation failed for {item_id}: {e}")
            return RecommendationResult.empty()
    
    async def update_user_feedback(self, user_id: str, item_id: str, 
                                 feedback: Dict[str, Any]) -> bool:
        """更新用户反馈"""
        try:
            # 更新用户画像
            await self._update_user_profile(user_id, item_id, feedback)
            
            # 更新物品画像
            await self._update_item_profile(item_id, feedback)
            
            # 清除相关缓存
            await self._clear_user_cache(user_id)
            
            logger.info(f"Updated feedback for user {user_id}, item {item_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update feedback: {e}")
            return False
    
    async def _get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """获取用户画像"""
        if user_id in self.user_profiles:
            return self.user_profiles[user_id]
        
        # 从数据库或外部服务加载
        profile = await self._load_user_profile_from_db(user_id)
        if profile:
            self.user_profiles[user_id] = profile
        
        return profile
    
    async def _get_item_profile(self, item_id: str) -> Optional[ItemProfile]:
        """获取物品画像"""
        if item_id in self.item_profiles:
            return self.item_profiles[item_id]
        
        # 从数据库或外部服务加载
        profile = await self._load_item_profile_from_db(item_id)
        if profile:
            self.item_profiles[item_id] = profile
        
        return profile
    
    async def _load_user_profile_from_db(self, user_id: str) -> Optional[UserProfile]:
        """从数据库加载用户画像"""
        # 简化实现 - 实际项目中连接真实数据库
        return UserProfile(
            user_id=user_id,
            preferences={"category": {"electronics": 0.8, "books": 0.6}},
            interaction_history=[]
        )
    
    async def _load_item_profile_from_db(self, item_id: str) -> Optional[ItemProfile]:
        """从数据库加载物品画像"""
        # 简化实现 - 实际项目中连接真实数据库
        return ItemProfile(
            item_id=item_id,
            features={"category": "electronics", "price_range": "medium"},
            content_vector=np.random.rand(100).tolist()
        )
    
    async def _handle_cold_start(self, user_id: str, top_k: int, 
                               context: Dict[str, Any]) -> RecommendationResult:
        """处理冷启动问题"""
        # 获取热门物品
        popular_items = await self._get_popular_items(top_k)
        
        return RecommendationResult(
            user_id=user_id,
            items=popular_items,
            scores=[0.8] * len(popular_items),
            algorithm_used="cold_start",
            context=context,
            generated_at=datetime.now()
        )
    
    async def _get_popular_items(self, count: int) -> List[str]:
        """获取热门物品"""
        # 简化实现 - 返回模拟热门物品
        return [f"popular_item_{i}" for i in range(count)]
    
    async def _post_process_recommendations(self, result: RecommendationResult,
                                          user_id: str, context: Dict[str, Any]) -> RecommendationResult:
        """推荐结果后处理"""
        # 去重
        seen_items = set()
        unique_items = []
        unique_scores = []
        
        for item_id, score in zip(result.items, result.scores):
            if item_id not in seen_items:
                unique_items.append(item_id)
                unique_scores.append(score)
                seen_items.add(item_id)
        
        # 多样性调整
        if len(unique_items) > 1:
            unique_items, unique_scores = self._ensure_diversity(
                unique_items, unique_scores, context
            )
        
        result.items = unique_items[:result.requested_count]
        result.scores = unique_scores[:result.requested_count]
        return result
    
    def _ensure_diversity(self, items: List[str], scores: List[float], 
                         context: Dict[str, Any]) -> Tuple[List[str], List[float]]:
        """确保推荐结果的多样性"""
        # 简化实现 - 实际项目中可以使用MMR算法
        return items, scores
    
    def _record_recommendation_history(self, user_id: str, result: RecommendationResult) -> None:
        """记录推荐历史"""
        self.recommendation_history[user_id].append({
            'timestamp': datetime.now(),
            'items': result.items,
            'algorithm': result.algorithm_used
        })
        
        # 限制历史记录长度
        if len(self.recommendation_history[user_id]) > 100:
            self.recommendation_history[user_id] = self.recommendation_history[user_id][-100:]
    
    def _update_stats(self, processing_time: float) -> None:
        """更新统计信息"""
        self.stats['total_recommendations'] += 1
        total = self.stats['total_recommendations']
        current_avg = self.stats['average_response_time']
        self.stats['average_response_time'] = (
            (current_avg * (total - 1) + processing_time) / total
        )
        self.stats['last_updated'] = datetime.now()
    
    async def _load_profiles(self) -> None:
        """加载用户和物品画像"""
        # 简化实现 - 实际项目中从数据库批量加载
        logger.info("Loading user and item profiles...")
    
    async def _warmup_cache(self) -> None:
        """预热缓存"""
        # 简化实现 - 加载热门推荐到缓存
        logger.info("Warming up recommendation cache...")
    
    async def _clear_user_cache(self, user_id: str) -> None:
        """清除用户相关缓存"""
        # 清除该用户的所有缓存键
        pattern = f"rec:{user_id}:*"
        await self.cache.clear_pattern(pattern)
    
    async def _update_user_profile(self, user_id: str, item_id: str, 
                                 feedback: Dict[str, Any]) -> None:
        """更新用户画像"""
        profile = await self._get_user_profile(user_id)
        if profile:
            profile.update_interaction(item_id, feedback)
            self.user_profiles[user_id] = profile
    
    async def _update_item_profile(self, item_id: str, 
                                 feedback: Dict[str, Any]) -> None:
        """更新物品画像"""
        profile = await self._get_item_profile(item_id)
        if profile:
            profile.update_feedback(feedback)
            self.item_profiles[item_id] = profile
    
    def get_stats(self) -> Dict[str, Any]:
        """获取推荐引擎统计"""
        return {
            **self.stats,
            'user_profiles_count': len(self.user_profiles),
            'item_profiles_count': len(self.item_profiles),
            'cache_stats': self.cache.get_stats(),
            'configuration': {
                'algorithms': ['collaborative', 'content_based', 'hybrid'],
                'cache_ttl': self.config['cache']['ttl']
            }
        }