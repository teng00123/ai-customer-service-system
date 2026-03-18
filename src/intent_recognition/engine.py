import logging
from typing import Dict, List, Tuple, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from ..core.config import settings

logger = logging.getLogger(__name__)

class IntentRecognizer:
    """意图识别引擎"""
    
    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        self.model = SentenceTransformer(model_name)
        self.intent_templates = {}
        self.intent_examples = {}
        self.is_trained = False
        
    def load_intents(self, intent_data: Dict[str, List[str]]) -> None:
        """加载意图模板和示例"""
        self.intent_examples = intent_data
        self.intent_templates = {}
        
        for intent, examples in intent_data.items():
            # 计算每个意图的平均向量作为模板
            embeddings = self.model.encode(examples)
            self.intent_templates[intent] = np.mean(embeddings, axis=0)
            
        self.is_trained = True
        logger.info(f"Loaded {len(intent_data)} intents")
    
    def recognize_intent(self, text: str, threshold: float = None) -> Tuple[str, float]:
        """识别文本意图"""
        if not self.is_trained:
            raise ValueError("Intent recognizer not trained. Load intents first.")
            
        if threshold is None:
            threshold = settings.similarity_threshold
            
        # 编码输入文本
        text_embedding = self.model.encode([text])
        
        best_intent = "unknown"
        best_score = 0.0
        
        # 计算与各意图模板的相似度
        for intent, template in self.intent_templates.items():
            similarity = cosine_similarity(text_embedding, template.reshape(1, -1))[0][0]
            if similarity > best_score:
                best_score = similarity
                best_intent = intent
                
        # 如果最高分低于阈值，返回unknown
        if best_score < threshold:
            return "unknown", best_score
            
        return best_intent, best_score
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """简单的实体抽取（基于关键词匹配）"""
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
                
        return entities