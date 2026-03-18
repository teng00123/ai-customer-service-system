"""AI引擎主控制器 - Claude Code风格"""
import logging
import asyncio
from typing import Dict, List, Any, Optional, Union
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import torch

from .nlp_processor import NLPProcessor
from .model_manager import ModelManager
from ..intent_recognition.models import ModelLoader
from ..recommendation_engine.engine import RecommendationEngine
from ..core.config import settings

logger = logging.getLogger(__name__)

class AIModelEngine:
    """AI模型引擎主控制器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._default_config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 核心组件
        self.model_manager = ModelManager(self.config.get('model_management', {}))
        self.nlp_processor = NLPProcessor(self.config.get('nlp', {}))
        self.model_loader = ModelLoader()
        self.recommendation_engine = RecommendationEngine(self.config.get('recommendation', {}))
        
        # 模型实例
        self.intent_model = None
        self.sentiment_model = None
        self.ner_model = None
        
        # 执行器
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # 统计信息
        self.stats = {
            'requests_processed': 0,
            'errors_count': 0,
            'average_response_time': 0.0,
            'last_updated': datetime.now()
        }
        
        logger.info(f"AI Engine initialized on device: {self.device}")
        
    def _default_config(self) -> Dict[str, Any]:
        """默认配置"""
        return {
            'model_management': {
                'auto_load': True,
                'model_cache_size': 3,
                'inference_timeout': 30
            },
            'nlp': {
                'enable_spell_check': True,
                'synonym_expansion': True,
                'max_text_length': 512
            },
            'recommendation': {
                'enabled': True,
                'default_n_recommendations': 5
            },
            'performance': {
                'batch_size': 32,
                'enable_profiling': False
            }
        }
    
    async def initialize(self) -> None:
        """异步初始化AI引擎"""
        try:
            logger.info("Initializing AI Engine components...")
            
            # 加载意图识别模型
            if self.config.get('model_management', {}).get('auto_load', True):
                await self._load_core_models()
            
            # 初始化推荐引擎
            if self.config.get('recommendation', {}).get('enabled', True):
                await self._initialize_recommendation_engine()
                
            logger.info("AI Engine initialization completed")
            
        except Exception as e:
            logger.error(f"AI Engine initialization failed: {e}")
            raise
    
    async def _load_core_models(self) -> None:
        """加载核心模型"""
        loop = asyncio.get_event_loop()
        
        # 加载意图识别模型
        self.intent_model = await loop.run_in_executor(
            self.executor, 
            self.model_loader.load_intent_model,
            settings.intent_model_path,
            10  # 默认10个意图类别
        )
        
        # 加载情感分析模型
        self.sentiment_model = await loop.run_in_executor(
            self.executor,
            self.model_loader.load_sentiment_model,
            settings.sentiment_model_path
        )
        
        logger.info("Core models loaded successfully")
    
    async def _initialize_recommendation_engine(self) -> None:
        """初始化推荐引擎"""
        # 添加一些示例数据用于演示
        sample_user = self.recommendation_engine.user_profiles.get('demo_user')
        if not sample_user:
            from ..recommendation_engine.models import UserProfile, ItemProfile
            demo_user = UserProfile(
                user_id='demo_user',
                interests=['手机', '电脑', '耳机'],
                preferences={'电子产品': 0.9, '配件': 0.7}
            )
            self.recommendation_engine.add_user_profile('demo_user', demo_user)
            
            # 添加示例物品
            sample_items = [
                ItemProfile('item1', 'iPhone 15', '最新款智能手机', '电子产品', 
                           ['手机', '苹果', '智能'], price=5999, popularity_score=0.8),
                ItemProfile('item2', 'MacBook Pro', '专业笔记本电脑', '电子产品',
                           ['电脑', '苹果', '笔记本'], price=12999, popularity_score=0.9),
                ItemProfile('item3', 'AirPods Pro', '主动降噪耳机', '配件',
                           ['耳机', '苹果', '无线'], price=1999, popularity_score=0.7)
            ]
            
            for item in sample_items:
                self.recommendation_engine.add_item_profile(item.item_id, item)
                
        logger.info("Recommendation engine initialized")
    
    async def process_text(self, text: str, task_type: str = "intent", 
                          context: Dict[str, Any] = None) -> Dict[str, Any]:
        """处理文本任务"""
        start_time = datetime.now()
        
        try:
            # NLP预处理
            processed_text = await self.nlp_processor.preprocess(text, task_type)
            
            # 根据任务类型调用相应模型
            if task_type == "intent":
                result = await self._predict_intent(processed_text, context)
            elif task_type == "sentiment":
                result = await self._predict_sentiment(processed_text)
            elif task_type == "ner":
                result = await self._extract_entities(processed_text)
            elif task_type == "comprehensive":
                result = await self._comprehensive_analysis(processed_text, context)
            else:
                raise ValueError(f"Unsupported task type: {task_type}")
            
            # 更新统计信息
            processing_time = (datetime.now() - start_time).total_seconds()
            self._update_stats(processing_time, success=True)
            
            return {
                'success': True,
                'task_type': task_type,
                'input_text': text,
                'processed_text': processed_text,
                'result': result,
                'processing_time': processing_time,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Text processing failed: {e}")
            self._update_stats(0, success=False)
            return {
                'success': False,
                'error': str(e),
                'task_type': task_type,
                'input_text': text,
                'timestamp': datetime.now().isoformat()
            }
    
    async def _predict_intent(self, text: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """预测意图"""
        if self.intent_model:
            intent, confidence = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self.model_loader.predict_intent,
                text,
                ["greeting", "goodbye", "order_query", "product_inquiry", 
                 "complaint", "refund", "technical_support", "payment", "delivery", "unknown"]
            )
        else:
            # 降级到规则匹配
            intent, confidence = self.model_loader._fallback_intent_prediction(text)
            
        return {
            'intent': intent,
            'confidence': confidence,
            'alternatives': []  # 备用意图列表
        }
    
    async def _predict_sentiment(self, text: str) -> Dict[str, Any]:
        """预测情感"""
        if self.sentiment_model:
            # 使用深度学习模型
            sentiment, confidence = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self.model_loader.predict_sentiment,
                text
            )
        else:
            # 使用简化规则
            sentiment, confidence = self.model_loader.predict_sentiment(text)
            
        return {
            'sentiment': sentiment,
            'confidence': confidence,
            'scores': {'positive': 0.0, 'negative': 0.0, 'neutral': 0.0}  # 简化版
        }
    
    async def _extract_entities(self, text: str) -> Dict[str, Any]:
        """抽取实体"""
        # 使用现有的实体抽取逻辑
        entities = {
            "product": [],
            "order_id": [],
            "date": [],
            "time": [],
            "location": []
        }
        
        # 简单的关键词匹配（实际项目中应使用NER模型）
        product_keywords = ["手机", "电脑", "耳机", "充电器", "配件"]
        order_patterns = ["订单", "单号", "编号"]
        
        for keyword in product_keywords:
            if keyword in text:
                entities["product"].append(keyword)
                
        for pattern in order_patterns:
            if pattern in text:
                # 尝试提取数字
                import re
                numbers = re.findall(r'\d{6,}', text)
                entities["order_id"].extend(numbers)
                
        return {
            'entities': entities,
            'confidence': 0.8,  # 固定置信度
            'model_version': 'rule_based_v1.0'
        }
    
    async def _comprehensive_analysis(self, text: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """综合分析"""
        # 并行执行多个任务
        intent_task = self._predict_intent(text, context)
        sentiment_task = self._predict_sentiment(text)
        entities_task = self._extract_entities(text)
        
        results = await asyncio.gather(intent_task, sentiment_task, entities_task)
        
        return {
            'intent_analysis': results[0],
            'sentiment_analysis': results[1],
            'entity_extraction': results[2],
            'context': context or {}
        }
    
    async def get_recommendations(self, user_id: str, n_recommendations: int = None,
                                 context: Dict[str, Any] = None) -> Dict[str, Any]:
        """获取推荐"""
        try:
            n_recs = n_recommendations or self.config.get('recommendation', {}).get('default_n_recommendations', 5)
            
            recommendations = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self.recommendation_engine.recommend_for_user,
                user_id, n_recs, context
            )
            
            return {
                'success': True,
                'user_id': user_id,
                'recommendations': recommendations,
                'count': len(recommendations),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Recommendation failed for user {user_id}: {e}")
            return {
                'success': False,
                'error': str(e),
                'user_id': user_id
            }
    
    def _update_stats(self, processing_time: float, success: bool) -> None:
        """更新统计信息"""
        self.stats['requests_processed'] += 1
        if not success:
            self.stats['errors_count'] += 1
            
        # 更新平均响应时间
        total_requests = self.stats['requests_processed']
        current_avg = self.stats['average_response_time']
        self.stats['average_response_time'] = (
            (current_avg * (total_requests - 1) + processing_time) / total_requests
        )
        
        self.stats['last_updated'] = datetime.now()
    
    def get_stats(self) -> Dict[str, Any]:
        """获取引擎统计信息"""
        return self.stats.copy()
    
    async def cleanup(self) -> None:
        """清理资源"""
        logger.info("Cleaning up AI Engine resources...")
        self.executor.shutdown(wait=True)
        logger.info("AI Engine cleanup completed")