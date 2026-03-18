"""AI模型定义和加载器"""
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification, AutoTokenizer, AutoModel
from typing import Dict, List, Tuple, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)

class IntentModel(nn.Module):
    """意图识别模型 (BERT-Chinese + 分类头)"""
    
    def __init__(self, model_name: str = "bert-base-chinese", num_labels: int = 10):
        super(IntentModel, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = outputs.logits
        return logits

class SentimentModel(nn.Module):
    """情感分析模型 (LSTM + Attention)"""
    
    def __init__(self, vocab_size: int, embed_dim: int = 300, hidden_dim: int = 256, num_classes: int = 3):
        super(SentimentModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, input_ids):
        embedded = self.embedding(input_ids)
        lstm_out, _ = self.lstm(embedded)
        
        # Attention机制
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        attended = torch.sum(attention_weights * lstm_out, dim=1)
        
        attended = self.dropout(attended)
        output = self.classifier(attended)
        return output

class NERModel(nn.Module):
    """命名实体识别模型 (BERT + CRF)"""
    
    def __init__(self, model_name: str = "bert-base-chinese", num_labels: int = 9):
        super(NERModel, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        return logits

class ModelLoader:
    """模型加载器 - Claude Code风格"""
    
    def __init__(self):
        self.models = {}
        self.tokenizers = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
    def load_intent_model(self, model_path: str, num_labels: int = 10) -> IntentModel:
        """加载意图识别模型"""
        try:
            tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
            model = IntentModel("bert-base-chinese", num_labels)
            
            # 尝试加载预训练权重
            try:
                model.load_state_dict(torch.load(model_path, map_location=self.device))
                logger.info(f"Loaded intent model from {model_path}")
            except FileNotFoundError:
                logger.warning(f"Model file {model_path} not found, using base BERT model")
                
            model.to(self.device)
            model.eval()
            
            self.models['intent'] = model
            self.tokenizers['intent'] = tokenizer
            return model
            
        except Exception as e:
            logger.error(f"Failed to load intent model: {e}")
            raise
            
    def load_sentiment_model(self, model_path: str, vocab_size: int = 10000) -> SentimentModel:
        """加载情感分析模型"""
        try:
            model = SentimentModel(vocab_size)
            
            try:
                model.load_state_dict(torch.load(model_path, map_location=self.device))
                logger.info(f"Loaded sentiment model from {model_path}")
            except FileNotFoundError:
                logger.warning(f"Model file {model_path} not found, initializing new model")
                
            model.to(self.device)
            model.eval()
            
            self.models['sentiment'] = model
            return model
            
        except Exception as e:
            logger.error(f"Failed to load sentiment model: {e}")
            raise
            
    def predict_intent(self, text: str, intent_labels: List[str]) -> Tuple[str, float]:
        """预测意图"""
        if 'intent' not in self.models:
            # 降级到规则匹配
            return self._fallback_intent_prediction(text)
            
        try:
            tokenizer = self.tokenizers['intent']
            model = self.models['intent']
            
            # 编码文本
            inputs = tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=128,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            
            input_ids = inputs['input_ids'].to(self.device)
            attention_mask = inputs['attention_mask'].to(self.device)
            
            # 预测
            with torch.no_grad():
                outputs = model(input_ids, attention_mask)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted_idx = torch.max(probabilities, dim=1)
                
            intent_label = intent_labels[predicted_idx.item()] if predicted_idx.item() < len(intent_labels) else "unknown"
            confidence_score = confidence.item()
            
            return intent_label, confidence_score
            
        except Exception as e:
            logger.error(f"Intent prediction failed: {e}")
            return self._fallback_intent_prediction(text)
            
    def predict_sentiment(self, text: str) -> Tuple[str, float]:
        """预测情感"""
        # 简化实现 - 实际应使用训练好的模型
        positive_words = ['好', '满意', '喜欢', '棒', '赞', '优秀']
        negative_words = ['差', '不好', '失望', '糟糕', '坏', '垃圾']
        
        pos_count = sum(1 for word in positive_words if word in text)
        neg_count = sum(1 for word in negative_words if word in text)
        
        if pos_count > neg_count:
            return "positive", min(0.9, 0.5 + pos_count * 0.2)
        elif neg_count > pos_count:
            return "negative", min(0.9, 0.5 + neg_count * 0.2)
        else:
            return "neutral", 0.7
            
    def _fallback_intent_prediction(self, text: str) -> Tuple[str, float]:
        """降级意图预测 (规则匹配)"""
        intent_keywords = {
            "greeting": ["你好", "您好", "嗨", "hello", "hi"],
            "goodbye": ["再见", "拜拜", "谢谢", "thank"],
            "order_query": ["订单", "查询", "状态", "物流"],
            "product_inquiry": ["产品", "价格", "规格", "功能"],
            "complaint": ["投诉", "不满", "问题", "糟糕"],
            "refund": ["退款", "退货", "换货", "取消"]
        }
        
        text_lower = text.lower()
        best_intent = "unknown"
        best_score = 0.0
        
        for intent, keywords in intent_keywords.items():
            score = sum(1 for keyword in keywords if keyword.lower() in text_lower)
            if score > best_score:
                best_score = score
                best_intent = intent
                
        confidence = min(0.8, best_score * 0.3) if best_score > 0 else 0.1
        return best_intent, confidence