"""基于内容的推荐算法 - Claude Code风格"""
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import jieba
import re
import logging

logger = logging.getLogger(__name__)

class ContentBasedFiltering:
    """基于内容的推荐算法"""
    
    def __init__(self, similarity_threshold: float = 0.3, max_features: int = 1000):
        self.similarity_threshold = similarity_threshold
        self.max_features = max_features
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words=['的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个'],
            ngram_range=(1, 2)
        )
        self.item_features = {}  # 物品特征矩阵
        self.item_ids = []  # 物品ID列表
        self.feature_names = []  # 特征名称
        
    def train(self, interactions_df: pd.DataFrame) -> None:
        """训练基于内容的模型"""
        logger.info("Training content-based filtering model...")
        
        # 构建物品特征
        self._build_item_features(interactions_df)
        
        # 如果有足够的文本数据，训练TF-IDF
        if self.item_features:
            self._train_tfidf()
            
        logger.info("Content-based filtering training completed")
    
    def _build_item_features(self, df: pd.DataFrame) -> None:
        """构建物品特征"""
        # 按物品分组，聚合交互信息
        item_stats = df.groupby('item_id').agg({
            'user_id': 'count',  # 交互次数
            'value': ['mean', 'std', 'sum'],  # 评分统计
            'interaction_type': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'click'
        }).fillna(0)
        
        # 展平列名
        item_stats.columns = ['_'.join(col).strip() for col in item_stats.columns.values]
        
        # 转换为特征字典
        for item_id, row in item_stats.iterrows():
            self.item_features[item_id] = {
                'interaction_count': row['user_id_count'],
                'avg_rating': row['value_mean'],
                'rating_std': row['value_std'],
                'total_value': row['value_sum'],
                'primary_interaction': row['interaction_type_<lambda>']
            }
    
    def _train_tfidf(self) -> None:
        """训练TF-IDF向量化器"""
        # 构建文本语料库
        documents = []
        self.item_ids = list(self.item_features.keys())
        
        for item_id in self.item_ids:
            features = self.item_features[item_id]
            # 将数值特征转换为文本描述
            text_parts = [
                f"interaction_{features.get('interaction_count', 0)}",
                f"rating_{features.get('avg_rating', 0):.1f}",
                f"type_{features.get('primary_interaction', 'unknown')}"
            ]
            documents.append(' '.join(text_parts))
        
        # 训练TF-IDF
        if documents:
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(documents)
            self.feature_names = self.tfidf_vectorizer.get_feature_names_out()
            
            # 保存物品特征向量
            for idx, item_id in enumerate(self.item_ids):
                self.item_features[item_id]['tfidf_vector'] = tfidf_matrix[idx]
    
    def recommend(self, user_id: str, user_profiles: Dict[str, Any], 
                 item_profiles: Dict[str, Any], n_recommendations: int) -> List[Dict[str, Any]]:
        """为用户生成基于内容的推荐"""
        user_profile = user_profiles.get(user_id)
        if not user_profile:
            # 冷启动：返回特征最丰富的物品
            return self._get_feature_rich_items(item_profiles, n_recommendations)
        
        # 基于用户兴趣匹配物品
        scored_items = []
        
        for item_id, item_profile in item_profiles.items():
            if item_id in self.item_features:
                score = self._calculate_content_similarity(user_profile, item_profile)
                if score >= self.similarity_threshold:
                    scored_items.append((item_id, score))
        
        # 排序并返回top-N
        scored_items.sort(key=lambda x: x[1], reverse=True)
        
        recommendations = []
        for rank, (item_id, score) in enumerate(scored_items[:n_recommendations]):
            recommendations.append({
                'item_id': item_id,
                'score': score,
                'rank': rank + 1,
                'method': 'content_based'
            })
            
        return recommendations
    
    def _calculate_content_similarity(self, user_profile, item_profile) -> float:
        """计算用户和物品的内容相似度"""
        score = 0.0
        
        # 兴趣标签匹配
        user_interests = set(user_profile.interests)
        item_tags = set(item_profile.tags)
        if user_interests and item_tags:
            interest_overlap = len(user_interests & item_tags)
            score += interest_overlap * 0.4
        
        # 类别偏好匹配
        if hasattr(user_profile, 'preferences') and item_profile.category:
            category_pref = user_profile.preferences.get(item_profile.category, 0.0)
            score += category_pref * 0.3
        
        # 价格偏好匹配（假设用户有价格区间偏好）
        if hasattr(user_profile, 'price_range') and item_profile.price > 0:
            if user_profile.price_range[0] <= item_profile.price <= user_profile.price_range[1]:
                score += 0.2
        
        # 物品流行度加成
        score += item_profile.popularity_score * 0.1
        
        return min(score, 1.0)
    
    def _get_feature_rich_items(self, item_profiles: Dict[str, Any], n: int) -> List[Dict[str, Any]]:
        """获取特征最丰富的物品"""
        scored_items = []
        
        for item_id, profile in item_profiles.items():
            richness_score = 0.0
            
            # 标签数量
            if profile.tags:
                richness_score += len(profile.tags) * 0.1
            
            # 特征丰富度
            if profile.features:
                richness_score += len(profile.features) * 0.05
            
            # 流行度
            richness_score += profile.popularity_score * 0.3
            
            scored_items.append((item_id, richness_score))
        
        scored_items.sort(key=lambda x: x[1], reverse=True)
        
        recommendations = []
        for rank, (item_id, score) in enumerate(scored_items[:n]):
            recommendations.append({
                'item_id': item_id,
                'score': score,
                'rank': rank + 1,
                'method': 'feature_rich'
            })
            
        return recommendations
    
    def extract_keywords(self, text: str, top_k: int = 10) -> List[str]:
        """提取关键词"""
        # 中文分词
        words = jieba.cut(text)
        
        # 过滤停用词和短词
        filtered_words = [
            word for word in words 
            if len(word) > 1 and word not in self.tfidf_vectorizer.stop_words_
        ]
        
        return filtered_words[:top_k]