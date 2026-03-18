"""推荐引擎模块 - Claude Code风格实现"""
from .engine import RecommendationEngine
from .models import UserProfile, ItemProfile, RecommendationResult
from .collaborative_filtering import CollaborativeFiltering
from .content_based import ContentBasedRecommender
from .hybrid import HybridRecommender

__all__ = [
    "RecommendationEngine",
    "UserProfile", 
    "ItemProfile",
    "RecommendationResult",
    "CollaborativeFiltering",
    "ContentBasedRecommender",
    "HybridRecommender"
]