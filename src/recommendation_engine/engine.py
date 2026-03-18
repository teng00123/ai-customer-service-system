"""推荐引擎主控制器 - Claude Code风格"""
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from .models import UserProfile, ItemProfile, RecommendationModel
from .collaborative_filtering import CollaborativeFiltering
from .content_based import ContentBasedFiltering

logger = logging.getLogger(__name__)

class RecommendationEngine:
    """智能推荐引擎"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._default_config()
        self.cf_engine = CollaborativeFiltering(**self.config.get('collaborative_filtering', {}))
        self.cb_engine = ContentBasedFiltering(**self.config.get('content_based', {}))
        self.user_profiles = {}  # 用户画像缓存
        self.item_profiles = {}  # 物品画像缓存
        self.interaction_history = []  # 交互历史
        
        logger.info("Recommendation engine initialized")
        
    def _default_config(self) -> Dict[str, Any]:
        """默认配置"""
        return {
            'collaborative_filtering': {
                'n_factors': 50,
                'learning_rate': 0.01,
                'regularization': 0.02,
                'epochs': 100
            },
            'content_based': {
                'similarity_threshold': 0.3,
                'max_features': 1000
            },
            'ensemble': {
                'cf_weight': 0.6,
                'cb_weight': 0.4
            }
        }
    
    def add_user_profile(self, user_id: str, profile: UserProfile) -> None:
        """添加用户画像"""
        self.user_profiles[user_id] = profile
        logger.info(f"Added user profile: {user_id}")
        
    def add_item_profile(self, item_id: str, profile: ItemProfile) -> None:
        """添加物品画像"""
        self.item_profiles[item_id] = profile
        logger.info(f"Added item profile: {item_id}")
        
    def record_interaction(self, user_id: str, item_id: str, 
                         interaction_type: str, value: float = 1.0) -> None:
        """记录用户交互"""
        interaction = {
            'user_id': user_id,
            'item_id': item_id,
            'interaction_type': interaction_type,  # click, purchase, rating, etc.
            'value': value,
            'timestamp': datetime.now()
        }
        self.interaction_history.append(interaction)
        
        # 更新用户画像
        if user_id in self.user_profiles:
            self.user_profiles[user_id].update_from_interaction(item_id, interaction_type, value)
            
        # 更新物品画像
        if item_id in self.item_profiles:
            self.item_profiles[item_id].update_from_interaction(user_id, interaction_type, value)
            
        logger.debug(f"Recorded interaction: {user_id} -> {item_id} ({interaction_type})")
    
    def recommend_for_user(self, user_id: str, n_recommendations: int = 5,
                          context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """为用户生成推荐"""
        try:
            if user_id not in self.user_profiles:
                logger.warning(f"User profile not found: {user_id}")
                return self._cold_start_recommendations(n_recommendations, context)
                
            # 协同过滤推荐
            cf_recommendations = self.cf_engine.recommend(
                user_id, self.user_profiles, self.item_profiles, n_recommendations * 2
            )
            
            # 基于内容的推荐
            cb_recommendations = self.cb_engine.recommend(
                user_id, self.user_profiles, self.item_profiles, n_recommendations * 2
            )
            
            # 融合推荐结果
            final_recommendations = self._ensemble_recommendations(
                cf_recommendations, cb_recommendations, n_recommendations
            )
            
            # 应用上下文过滤
            if context:
                final_recommendations = self._apply_context_filter(final_recommendations, context)
                
            logger.info(f"Generated {len(final_recommendations)} recommendations for user {user_id}")
            return final_recommendations[:n_recommendations]
            
        except Exception as e:
            logger.error(f"Recommendation failed for user {user_id}: {e}")
            return []
    
    def _ensemble_recommendations(self, cf_recs: List[Dict], cb_recs: List[Dict], 
                               n_final: int) -> List[Dict]:
        """融合协同过滤和内容推荐结果"""
        ensemble_config = self.config.get('ensemble', {'cf_weight': 0.6, 'cb_weight': 0.4})
        cf_weight = ensemble_config.get('cf_weight', 0.6)
        cb_weight = ensemble_config.get('cb_weight', 0.4)
        
        # 创建物品得分映射
        item_scores = {}
        
        # 处理协同过滤结果
        for rec in cf_recs:
            item_id = rec['item_id']
            score = rec.get('score', 0.5) * cf_weight
            item_scores[item_id] = item_scores.get(item_id, 0) + score
            
        # 处理基于内容的结果
        for rec in cb_recs:
            item_id = rec['item_id']
            score = rec.get('score', 0.5) * cb_weight
            item_scores[item_id] = item_scores.get(item_id, 0) + score
            
        # 排序并返回top-N
        sorted_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
        
        recommendations = []
        for item_id, score in sorted_items[:n_final]:
            item_profile = self.item_profiles.get(item_id)
            recommendations.append({
                'item_id': item_id,
                'score': min(score, 1.0),
                'reason': self._generate_recommendation_reason(user_id, item_id),
                'item_info': item_profile.to_dict() if item_profile else {}
            })
            
        return recommendations
    
    def _cold_start_recommendations(self, n_recommendations: int, 
                                   context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """冷启动推荐策略"""
        # 热门物品推荐
        popular_items = sorted(
            self.item_profiles.items(),
            key=lambda x: x[1].popularity_score,
            reverse=True
        )[:n_recommendations]
        
        recommendations = []
        for item_id, profile in popular_items:
            recommendations.append({
                'item_id': item_id,
                'score': profile.popularity_score,
                'reason': 'popular_item',
                'item_info': profile.to_dict()
            })
            
        return recommendations
    
    def _apply_context_filter(self, recommendations: List[Dict], 
                            context: Dict[str, Any]) -> List[Dict]:
        """应用上下文过滤"""
        filtered = []
        for rec in recommendations:
            # 时间上下文
            if 'time_of_day' in context:
                # 根据时间段调整推荐
                pass
                
            # 设备上下文
            if 'device' in context:
                # 根据设备类型调整推荐
                pass
                
            filtered.append(rec)
            
        return filtered
    
    def _generate_recommendation_reason(self, user_id: str, item_id: str) -> str:
        """生成推荐理由"""
        user_profile = self.user_profiles.get(user_id)
        item_profile = self.item_profiles.get(item_id)
        
        if not user_profile or not item_profile:
            return "基于您的兴趣推荐"
            
        # 基于共同特征生成理由
        common_tags = set(user_profile.interests) & set(item_profile.tags)
        if common_tags:
            return f"因为您对{', '.join(list(common_tags)[:2])}感兴趣"
            
        return "基于相似用户的行为推荐"
    
    def train_models(self, training_data: pd.DataFrame = None) -> None:
        """训练推荐模型"""
        logger.info("Starting model training...")
        
        if training_data is not None:
            # 使用提供的训练数据
            self.cf_engine.train(training_data)
            self.cb_engine.train(training_data)
        else:
            # 使用交互历史训练
            if len(self.interaction_history) > 10:
                df = pd.DataFrame(self.interaction_history)
                self.cf_engine.train(df)
                self.cb_engine.train(df)
                
        logger.info("Model training completed")