"""推荐引擎数据模型 - Claude Code风格"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
import json

@dataclass
class UserProfile:
    """用户画像模型"""
    user_id: str
    interests: List[str] = field(default_factory=list)
    preferences: Dict[str, float] = field(default_factory=dict)  # 类别偏好权重
    behavior_pattern: Dict[str, int] = field(default_factory=dict)  # 行为模式统计
    demographic_info: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    popularity_score: float = 0.0
    
    def update_from_interaction(self, item_id: str, interaction_type: str, value: float):
        """从交互中更新用户画像"""
        # 更新行为模式
        self.behavior_pattern[interaction_type] = self.behavior_pattern.get(interaction_type, 0) + 1
        
        # 更新流行度评分（简化的用户活跃度）
        self.popularity_score = min(1.0, self.popularity_score + 0.01)
        self.last_updated = datetime.now()
        
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'user_id': self.user_id,
            'interests': self.interests,
            'preferences': self.preferences,
            'behavior_pattern': self.behavior_pattern,
            'demographic_info': self.demographic_info,
            'created_at': self.created_at.isoformat(),
            'last_updated': self.last_updated.isoformat(),
            'popularity_score': self.popularity_score
        }

@dataclass
class ItemProfile:
    """物品画像模型"""
    item_id: str
    name: str = ""
    description: str = ""
    category: str = ""
    tags: List[str] = field(default_factory=list)
    features: Dict[str, float] = field(default_factory=dict)  # 特征向量
    price: float = 0.0
    popularity_score: float = 0.0
    rating_average: float = 0.0
    rating_count: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    
    def update_from_interaction(self, user_id: str, interaction_type: str, value: float):
        """从交互中更新物品画像"""
        if interaction_type == 'rating':
            # 更新评分统计
            old_total = self.rating_average * self.rating_count
            self.rating_count += 1
            self.rating_average = (old_total + value) / self.rating_count
        elif interaction_type in ['purchase', 'click']:
            # 更新流行度
            self.popularity_score = min(1.0, self.popularity_score + 0.05)
            
        self.last_updated = datetime.now()
        
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'item_id': self.item_id,
            'name': self.name,
            'description': self.description,
            'category': self.category,
            'tags': self.tags,
            'features': self.features,
            'price': self.price,
            'popularity_score': self.popularity_score,
            'rating_average': self.rating_average,
            'rating_count': self.rating_count,
            'created_at': self.created_at.isoformat(),
            'last_updated': self.last_updated.isoformat()
        }

@dataclass
class RecommendationModel:
    """推荐模型配置"""
    model_id: str
    model_type: str  # collaborative_filtering, content_based, hybrid
    parameters: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    version: str = "1.0.0"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'model_id': self.model_id,
            'model_type': self.model_type,
            'parameters': self.parameters,
            'performance_metrics': self.performance_metrics,
            'created_at': self.created_at.isoformat(),
            'version': self.version
        }

@dataclass
class RecommendationResult:
    """推荐结果"""
    item_id: str
    score: float
    rank: int
    reason: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'item_id': self.item_id,
            'score': self.score,
            'rank': self.rank,
            'reason': self.reason,
            'metadata': self.metadata
        }