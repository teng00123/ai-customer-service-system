"""推荐系统API接口 - Claude Code风格"""
import logging
from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Dict, Any, Optional
from datetime import datetime

from ..recommendation.engine import RecommendationEngine
from ..recommendation.models import RecommendationResult
from ..core.config import settings
from ..core.dependencies import get_recommendation_engine

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/recommendations", tags=["recommendations"])

@router.post("/user/{user_id}", response_model=Dict[str, Any])
async def get_user_recommendations(
    user_id: str,
    algorithm: str = Query("hybrid", description="推荐算法: hybrid, collaborative, content_based"),
    top_k: int = Query(10, ge=1, le=50, description="推荐数量"),
    context: Optional[Dict[str, Any]] = None,
    engine: RecommendationEngine = Depends(get_recommendation_engine)
):
    """获取用户推荐"""
    try:
        result = await engine.recommend_for_user(
            user_id=user_id,
            algorithm=algorithm,
            top_k=top_k,
            context=context
        )
        
        return {
            "success": True,
            "data": result.to_dict(),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"User recommendation failed for {user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/item/{item_id}/similar", response_model=Dict[str, Any])
async def get_similar_items(
    item_id: str,
    top_k: int = Query(10, ge=1, le=50, description="相似物品数量"),
    context: Optional[Dict[str, Any]] = None,
    engine: RecommendationEngine = Depends(get_recommendation_engine)
):
    """获取相似物品推荐"""
    try:
        result = await engine.recommend_for_item(
            item_id=item_id,
            top_k=top_k,
            context=context
        )
        
        return {
            "success": True,
            "data": result.to_dict(),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Similar items recommendation failed for {item_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/feedback", response_model=Dict[str, Any])
async def submit_feedback(
    user_id: str,
    item_id: str,
    feedback: Dict[str, Any],
    engine: RecommendationEngine = Depends(get_recommendation_engine)
):
    """提交用户反馈"""
    try:
        success = await engine.update_user_feedback(user_id, item_id, feedback)
        
        if success:
            return {
                "success": True,
                "message": "Feedback updated successfully",
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=400, detail="Failed to update feedback")
            
    except Exception as e:
        logger.error(f"Feedback submission failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stats", response_model=Dict[str, Any])
async def get_recommendation_stats(
    engine: RecommendationEngine = Depends(get_recommendation_engine)
):
    """获取推荐引擎统计信息"""
    try:
        stats = engine.get_stats()
        return {
            "success": True,
            "data": stats,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get recommendation stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))