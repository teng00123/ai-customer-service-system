"""协同过滤算法实现 - Claude Code风格"""
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from scipy.sparse.linalg import svds
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

class CollaborativeFiltering:
    """协同过滤推荐算法"""
    
    def __init__(self, n_factors: int = 50, learning_rate: float = 0.01, 
                 regularization: float = 0.02, epochs: int = 100):
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.epochs = epochs
        self.user_factors = {}  # 用户因子矩阵
        self.item_factors = {}  # 物品因子矩阵
        self.user_mapping = {}  # 用户ID映射
        self.item_mapping = {}  # 物品ID映射
        self.global_mean = 0.0
        
    def train(self, interactions_df: pd.DataFrame) -> None:
        """训练协同过滤模型"""
        logger.info("Training collaborative filtering model...")
        
        # 构建用户-物品交互矩阵
        user_item_matrix, user_ids, item_ids = self._build_interaction_matrix(interactions_df)
        self.global_mean = interactions_df['value'].mean()
        
        # 创建ID映射
        self.user_mapping = {uid: idx for idx, uid in enumerate(user_ids)}
        self.item_mapping = {iid: idx for idx, iid in enumerate(item_ids)}
        
        if len(user_ids) > self.n_factors and len(item_ids) > self.n_factors:
            # 使用SVD进行矩阵分解（适用于大矩阵）
            self._train_svd(user_item_matrix)
        else:
            # 使用梯度下降（适用于小矩阵）
            self._train_gradient_descent(user_item_matrix, user_ids, item_ids)
            
        logger.info("Collaborative filtering training completed")
    
    def _build_interaction_matrix(self, df: pd.DataFrame) -> tuple:
        """构建用户-物品交互矩阵"""
        # 透视表构建稀疏矩阵
        matrix = df.pivot_table(
            index='user_id', columns='item_id', values='value', fill_value=0
        )
        return matrix.values, list(matrix.index), list(matrix.columns)
    
    def _train_svd(self, matrix: np.ndarray) -> None:
        """使用SVD进行矩阵分解"""
        # 添加正则化的SVD
        U, sigma, Vt = svds(matrix.astype(np.float32), k=self.n_factors)
        sigma = np.diag(sigma)
        
        # 用户和物品隐向量
        self.user_factors = {str(i): U[i] @ sigma for i in range(U.shape[0])}
        self.item_factors = {str(j): Vt.T[j] @ sigma for j in range(Vt.shape[0])}
    
    def _train_gradient_descent(self, matrix: np.ndarray, user_ids: List[str], item_ids: List[str]) -> None:
        """使用梯度下降训练矩阵分解"""
        n_users, n_items = matrix.shape
        
        # 初始化隐向量
        np.random.seed(42)
        user_factors = np.random.normal(scale=1./self.n_factors, size=(n_users, self.n_factors))
        item_factors = np.random.normal(scale=1./self.n_factors, size=(n_items, self.n_factors))
        
        # 梯度下降训练
        for epoch in range(self.epochs):
            total_loss = 0.0
            
            for i in range(n_users):
                for j in range(n_items):
                    if matrix[i, j] > 0:  # 只处理有交互的项
                        # 预测评分
                        pred = np.dot(user_factors[i], item_factors[j])
                        error = matrix[i, j] - pred
                        total_loss += error ** 2
                        
                        # 更新隐向量
                        user_factors[i] += self.learning_rate * (
                            error * item_factors[j] - self.regularization * user_factors[i]
                        )
                        item_factors[j] += self.learning_rate * (
                            error * user_factors[i] - self.regularization * item_factors[j]
                        )
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch + 1}/{self.epochs}, Loss: {total_loss:.4f}")
        
        # 保存结果
        self.user_factors = {user_ids[i]: user_factors[i] for i in range(n_users)}
        self.item_factors = {item_ids[j]: item_factors[j] for j in range(n_items)}
    
    def recommend(self, user_id: str, user_profiles: Dict[str, Any], 
                 item_profiles: Dict[str, Any], n_recommendations: int) -> List[Dict[str, Any]]:
        """为用户生成推荐"""
        if user_id not in self.user_factors:
            # 冷启动：返回热门物品
            return self._get_popular_items(item_profiles, n_recommendations)
            
        user_vector = self.user_factors[user_id]
        scores = {}
        
        # 计算用户对所有物品的预测评分
        for item_id, item_vector in self.item_factors.items():
            if item_id in item_profiles:  # 只推荐存在的物品
                pred_score = np.dot(user_vector, item_vector)
                # 添加物品流行度加权
                popularity_bonus = item_profiles[item_id].popularity_score * 0.1
                scores[item_id] = pred_score + popularity_bonus
        
        # 排序并返回top-N
        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        recommendations = []
        for rank, (item_id, score) in enumerate(sorted_items[:n_recommendations]):
            recommendations.append({
                'item_id': item_id,
                'score': min(max(score, 0.0), 1.0),  # 归一化到0-1
                'rank': rank + 1,
                'method': 'collaborative_filtering'
            })
            
        return recommendations
    
    def _get_popular_items(self, item_profiles: Dict[str, Any], n: int) -> List[Dict[str, Any]]:
        """获取热门物品推荐"""
        sorted_items = sorted(
            item_profiles.items(),
            key=lambda x: x[1].popularity_score,
            reverse=True
        )
        
        recommendations = []
        for rank, (item_id, profile) in enumerate(sorted_items[:n]):
            recommendations.append({
                'item_id': item_id,
                'score': profile.popularity_score,
                'rank': rank + 1,
                'method': 'popular_items'
            })
            
        return recommendations