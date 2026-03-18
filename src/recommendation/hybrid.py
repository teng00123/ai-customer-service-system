"""混合推荐算法 - Claude Code风格"""
import logging
from typing import List, Dict, Any, Optional
from collections import defaultdict

from .models import UserProfile, ItemProfile, RecommendationResult
from .collaborative_filtering import CollaborativeFiltering
from .content_based import ContentBasedRecommender

logger = logging.getLogger(__name__)

class HybridRecommender:
    """混合推荐算法 - 结合协同过滤和基于内容的方法"""
    
    def __init__(self, config: Dict[str, Any], cf_engine: CollaborativeFiltering, cb_engine: ContentBasedRecommender):
        self.config = config
        self.cf_engine = cf_engine
        self.cb_engine = cb_engine
        
        # 混合权重
        self.cf_weight = config.get('cf_weight', 0.6)
        self.cb_weight = config.get('cb_weight', 0.4)
        self.diversity_factor = config.get('diversity_factor', 0.1)
        
        # 算法切换策略
        self.strategy = config.get('strategy', 'weighted')  # weighted, switching, cascade
        self.cf_threshold = config.get('cf_threshold', 0.3)  # 协同过滤最小相似用户数
        self.cb_threshold = config.get('cb_threshold', 0.2)  # 基于内容最小相似度
        
        # 统计信息
        self.stats = {
            'recommendations_generated': 0,
            'cf_used': 0,
            'cb_used': 0,
            'hybrid_used': 0,
            'strategy_switches': 0
        }
        
        logger.info(f"Hybrid recommender initialized with strategy: {self.strategy}")
    
    async def recommend(self, user_profile: UserProfile, top_k: int, 
                       context: Dict[str, Any] = None) -> RecommendationResult:
        """生成混合推荐"""
        try:
            user_id = user_profile.user_id
            
            if self.strategy == 'weighted':
                result = await self._weighted_hybrid(user_profile, top_k, context)
            elif self.strategy == 'switching':
                result = await self._switching_hybrid(user_profile, top_k, context)
            elif self.strategy == 'cascade':
                result = await self._cascade_hybrid(user_profile, top_k, context)
            else:
                logger.warning(f"Unknown strategy: {self.strategy}, falling back to weighted")
                result = await self._weighted_hybrid(user_profile, top_k, context)
            
            self.stats['recommendations_generated'] += 1
            return result
            
        except Exception as e:
            logger.error(f"Hybrid recommendation failed: {e}")
            return RecommendationResult.empty(user_profile.user_id)
    
    async def _weighted_hybrid(self, user_profile: UserProfile, top_k: int, 
                             context: Dict[str, Any] = None) -> RecommendationResult:
        """加权混合策略"""
        # 并行获取两种算法的推荐结果
        cf_task = self.cf_engine.recommend(user_profile, top_k, context)
        cb_task = self.cb_engine.recommend(user_profile, top_k, context)
        
        cf_result, cb_result = await asyncio.gather(cf_task, cb_task, return_exceptions=True)
        
        if isinstance(cf_result, Exception) or not cf_result.items:
            cf_result = RecommendationResult.empty(user_profile.user_id)
        if isinstance(cb_result, Exception) or not cb_result.items:
            cb_result = RecommendationResult.empty(user_profile.user_id)
        
        # 合并推荐结果
        merged_items = defaultdict(lambda: {'cf_score': 0.0, 'cb_score': 0.0, 'sources': []})
        
        # 添加协同过滤结果
        for item_id, cf_score in zip(cf_result.items, cf_result.scores):
            merged_items[item_id]['cf_score'] = cf_score
            merged_items[item_id]['sources'].append('cf')
        
        # 添加基于内容结果
        for item_id, cb_score in zip(cb_result.items, cb_result.scores):
            merged_items[item_id]['cb_score'] = cb_score
            merged_items[item_id]['sources'].append('cb')
        
        # 计算加权分数
        final_items = []
        final_scores = []
        
        for item_id, data in merged_items.items():
            # 加权组合分数
            cf_score = data['cf_score']
            cb_score = data['cb_score']
            
            # 如果某个算法没有提供分数，使用默认值
            if cf_score == 0.0 and 'cf' in data['sources']:
                cf_score = 0.5  # 默认中等分数
            if cb_score == 0.0 and 'cb' in data['sources']:
                cb_score = 0.5
            
            # 计算最终分数
            if 'cf' in data['sources'] and 'cb' in data['sources']:
                # 两个算法都推荐了
                final_score = self.cf_weight * cf_score + self.cb_weight * cb_score
                self.stats['hybrid_used'] += 1
            elif 'cf' in data['sources']:
                # 只有协同过滤推荐
                final_score = cf_score
                self.stats['cf_used'] += 1
            else:
                # 只有基于内容推荐
                final_score = cb_score
                self.stats['cb_used'] += 1
            
            # 多样性调整
            diversity_penalty = self._calculate_diversity_penalty(item_id, final_items)
            final_score *= (1.0 - self.diversity_factor * diversity_penalty)
            
            final_items.append(item_id)
            final_scores.append(final_score)
        
        # 按分数排序
        sorted_results = sorted(zip(final_items, final_scores), key=lambda x: x[1], reverse=True)
        
        items = [item_id for item_id, _ in sorted_results[:top_k]]
        scores = [score for _, score in sorted_results[:top_k]]
        
        return RecommendationResult(
            user_id=user_profile.user_id,
            items=items,
            scores=scores,
            algorithm_used="hybrid_weighted",
            context=context,
            requested_count=top_k
        )
    
    async def _switching_hybrid(self, user_profile: UserProfile, top_k: int, 
                              context: Dict[str, Any] = None) -> RecommendationResult:
        """切换策略 - 根据条件选择算法"""
        user_id = user_profile.user_id
        
        # 检查用户交互历史
        interaction_count = len(user_profile.interaction_history)
        
        if interaction_count < 5:  # 新用户，使用基于内容
            self.stats['strategy_switches'] += 1
            logger.debug(f"Switching to content-based for new user {user_id}")
            result = await self.cb_engine.recommend(user_profile, top_k, context)
            result.algorithm_used = "hybrid_switching_cb"
            self.stats['cb_used'] += 1
        else:
            # 尝试协同过滤
            cf_result = await self.cf_engine.recommend(user_profile, top_k, context)
            
            if len(cf_result.items) >= self.cf_threshold * top_k:
                # 协同过滤有足够结果
                result = cf_result
                result.algorithm_used = "hybrid_switching_cf"
                self.stats['cf_used'] += 1
            else:
                # 协同过滤结果不足，切换到基于内容
                self.stats['strategy_switches'] += 1
                logger.debug(f"Switching to content-based for user {user_id} due to insufficient CF results")
                result = await self.cb_engine.recommend(user_profile, top_k, context)
                result.algorithm_used = "hybrid_switching_cb"
                self.stats['cb_used'] += 1
        
        return result
    
    async def _cascade_hybrid(self, user_profile: UserProfile, top_k: int, 
                            context: Dict[str, Any] = None) -> RecommendationResult:
        """级联策略 - 先用一个算法，再用另一个算法补充"""
        # 第一步：使用协同过滤生成候选集
        cf_result = await self.cf_engine.recommend(user_profile, top_k * 2, context)
        
        if not cf_result.items:
            # 协同过滤失败，回退到基于内容
            result = await self.cb_engine.recommend(user_profile, top_k, context)
            result.algorithm_used = "hybrid_cascade_cb_fallback"
            self.stats['cb_used'] += 1
            return result
        
        # 第二步：用基于内容对候选集重新排序
        cb_scores = {}
        cb_result = await self.cb_engine.recommend(user_profile, top_k * 2, context)
        
        for item_id, score in zip(cb_result.items, cb_result.scores):
            cb_scores[item_id] = score
        
        # 组合分数
        final_items = []
        final_scores = []
        
        for item_id, cf_score in zip(cf_result.items, cf_result.scores):
            cb_score = cb_scores.get(item_id, 0.0)
            combined_score = 0.7 * cf_score + 0.3 * cb_score  # 侧重协同过滤
            
            final_items.append(item_id)
            final_scores.append(combined_score)
        
        # 排序并取前K个
        sorted_results = sorted(zip(final_items, final_scores), key=lambda x: x[1], reverse=True)
        items = [item_id for item_id, _ in sorted_results[:top_k]]
        scores = [score for _, score in sorted_results[:top_k]]
        
        result = RecommendationResult(
            user_id=user_profile.user_id,
            items=items,
            scores=scores,
            algorithm_used="hybrid_cascade",
            context=context,
            requested_count=top_k
        )
        
        self.stats['hybrid_used'] += 1
        return result
    
    def _calculate_diversity_penalty(self, item_id: str, existing_items: List[str]) -> float:
        """计算多样性惩罚因子"""
        if not existing_items:
            return 0.0
        
        # 简化的多样性计算 - 检查类别重复
        # 实际项目中可以使用更复杂的多样性度量
        penalty = 0.0
        for existing_item in existing_items[-3:]:  # 只看最近3个物品
            if self._same_category(item_id, existing_item):
                penalty += 0.3
        
        return min(penalty, 1.0)
    
    def _same_category(self, item1_id: str, item2_id: str) -> bool:
        """检查两个物品是否同类别 (简化实现)"""
        # 实际项目中应该从物品画像中获取类别信息
        return item1_id.split('_')[0] == item2_id.split('_')[0]
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            **self.stats,
            'cf_engine_stats': self.cf_engine.get_stats(),
            'cb_engine_stats': self.cb_engine.get_stats(),
            'configuration': {
                'strategy': self.strategy,
                'cf_weight': self.cf_weight,
                'cb_weight': self.cb_weight,
                'diversity_factor': self.diversity_factor
            }
        }