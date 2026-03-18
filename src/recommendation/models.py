"""推荐引擎数据模型 - Claude Code风格"""
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import numpy as np

class RecommendationAlgorithm(Enum):
    """推荐算法枚举"""
    COLLABORATIVE_FILTERING = "collaborative_filtering"
    CONTENT_BASED = "content_based"
    HYBRID = "hybrid"
    POPULARITY = "popularity"
    RANDOM = "random"

@dataclass
class UserProfile:
    """用户画像模型"""
    user_id: str
    preferences: Dict[str, float] = field(default_factory=dict)  # 类别偏好权重
    interaction_history: List[Dict[str, Any]] = field(default_factory=list)
    demographic_info: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def update_interaction(self, item_id: str, feedback: Dict[str, Any]) -> None:
        """更新用户交互历史"""
        interaction = {
            'item_id': item_id,
            'timestamp': datetime.now(),
            'feedback': feedback
        }
        self.interaction_history.append(interaction)
        self.updated_at = datetime.now()
        
        # 更新偏好权重
        rating = feedback.get('rating', 3.0)
        category = feedback.get('category', 'general')
        
        if category not in self.preferences:
            self.preferences[category] = 0.5
            
        # 指数移动平均更新偏好
        alpha = 0.1
        current_pref = self.preferences[category]
        new_pref = current_pref + alpha * ((rating - 3.0) / 2.0 - current_pref)
        self.preferences[category] = max(0.0, min(1.0, new_pref))
    
    def get_recent_interactions(self, days: int = 30) -> List[Dict[str, Any]]:
        """获取近期交互记录"""
        cutoff_date = datetime.now() - timedelta(days=days)
        return [
            interaction for interaction in self.interaction_history
            if interaction['timestamp'] >= cutoff_date
        ]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'user_id': self.user_id,
            'preferences': self.preferences,
            'interaction_history': self.interaction_history,
            'demographic_info': self.demographic_info,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UserProfile':
        return cls(
            user_id=data['user_id'],
            preferences=data.get('preferences', {}),
            interaction_history=data.get('interaction_history', []),
            demographic_info=data.get('demographic_info', {}),
            created_at=datetime.fromisoformat(data['created_at']) if data.get('created_at') else datetime.now(),
            updated_at=datetime.fromisoformat(data['updated_at']) if data.get('updated_at') else datetime.now()
        )

@dataclass
class ItemProfile:
    """物品画像模型"""
    item_id: str
    features: Dict[str, Any] = field(default_factory=dict)
    content_vector: List[float] = field(default_factory=list)
    category: str = "general"
    popularity_score: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def update_feedback(self, feedback: Dict[str, Any]) -> None:
        """更新物品反馈信息"""
        # 更新受欢迎度分数
        rating = feedback.get('rating', 3.0)
        self.popularity_score = 0.9 * self.popularity_score + 0.1 * (rating / 5.0)
        self.updated_at = datetime.now()
        
        # 更新特征
        for key, value in feedback.get('features', {}).items():
            self.features[key] = value
    
    def calculate_similarity(self, other: 'ItemProfile') -> float:
        """计算与其他物品的相似度"""
        if self.content_vector and other.content_vector:
            # 向量余弦相似度
            v1 = np.array(self.content_vector)
            v2 = np.array(other.content_vector)
            dot_product = np.dot(v1, v2)
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            
            if norm1 > 0 and norm2 > 0:
                return float(dot_product / (norm1 * norm2))
        
        # 基于特征的相似度
        common_features = set(self.features.keys()) & set(other.features.keys())
        if common_features:
            similarity = sum(
                1.0 for f in common_features 
                if self.features[f] == other.features[f]
            ) / len(common_features)
            return similarity
            
        return 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'item_id': self.item_id,
            'features': self.features,
            'content_vector': self.content_vector,
            'category': self.category,
            'popularity_score': self.popularity_score,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ItemProfile':
        return cls(
            item_id=data['item_id'],
            features=data.get('features', {}),
            content_vector=data.get('content_vector', []),
            category=data.get('category', 'general'),
            popularity_score=data.get('popularity_score', 0.0),
            created_at=datetime.fromisoformat(data['created_at']) if data.get('created_at') else datetime.now(),
            updated_at=datetime.fromisoformat(data['updated_at']) if data.get('updated_at') else datetime.now()
        )

@dataclass
class RecommendationResult:
    """推荐结果模型"""
    user_id: str
    items: List[str] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)
    algorithm_used: str = "unknown"
    context: Dict[str, Any] = field(default_factory=dict)
    generated_at: datetime = field(default_factory=datetime.now)
    requested_count: int = 10
    
    @property
    def item_scores(self) -> Dict[str, float]:
        """获取物品-分数映射"""
        return dict(zip(self.items, self.scores))
    
    def add_item(self, item_id: str, score: float) -> None:
        """添加推荐物品"""
        self.items.append(item_id)
        self.scores.append(score)
    
    def filter_by_score(self, min_score: float) -> None:
        """根据最低分数过滤"""
        filtered_items = []
        filtered_scores = []
        
        for item, score in zip(self.items, self.scores):
            if score >= min_score:
                filtered_items.append(item)
                filtered_scores.append(score)
                
        self.items = filtered_items
        self.scores = filtered_scores
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'user_id': self.user_id,
            'items': self.items,
            'scores': self.scores,
            'algorithm_used': self.algorithm_used,
            'context': self.context,
            'generated_at': self.generated_at.isoformat(),
            'requested_count': self.requested_count
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RecommendationResult':
        return cls(
            user_id=data['user_id'],
            items=data.get('items', []),
            scores=data.get('scores', []),
            algorithm_used=data.get('algorithm_used', 'unknown'),
            context=data.get('context', {}),
            generated_at=datetime.fromisoformat(data['generated_at']) if data.get('generated_at') else datetime.now(),
            requested_count=data.get('requested_count', 10)
        )
    
    @classmethod
    def empty(cls, user_id: str = "") -> 'RecommendationResult':
        """创建空推荐结果"""
        return cls(user_id=user_id)