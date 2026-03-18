"""对话上下文模型 - Claude Code风格"""
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

class DialogContext:
    """对话上下文管理器"""
    
    def __init__(self, session_id: str, user_id: str, max_history: int = 10):
        self.session_id = session_id
        self.user_id = user_id
        self.max_history = max_history
        
        # 上下文数据
        self.conversation_history = deque(maxlen=max_history)
        self.user_preferences = defaultdict(str)
        self.current_topic = ""
        self.entity_memory = defaultdict(list)  # 实体记忆
        self.emotion_state = {"sentiment": "neutral", "confidence": 0.5}
        self.session_goals = []
        self.resolved_issues = []
        
        # 上下文统计
        self.total_messages = 0
        self.last_update = datetime.now()
        
        logger.info(f"Dialog context initialized for session: {session_id}")
        
    def update_context(self, user_id: str, message: str, message_type: str = "text") -> None:
        """更新对话上下文"""
        try:
            # 添加消息到历史
            self.conversation_history.append({
                'user_id': user_id,
                'message': message,
                'type': message_type,
                'timestamp': datetime.now()
            })
            
            # 更新统计
            self.total_messages += 1
            self.last_update = datetime.now()
            
            # 提取主题（简化实现）
            self._extract_topic(message)
            
        except Exception as e:
            logger.error(f"Context update failed: {e}")
    
    def set_emotion(self, emotion_result: Dict[str, Any]) -> None:
        """设置情感状态"""
        try:
            sentiment = emotion_result.get('sentiment', 'neutral')
            confidence = emotion_result.get('confidence', 0.5)
            
            self.emotion_state = {
                "sentiment": sentiment,
                "confidence": confidence,
                "last_updated": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Emotion setting failed: {e}")
    
    def add_entity(self, entity_type: str, entity_value: str, confidence: float = 1.0) -> None:
        """添加实体到记忆"""
        try:
            self.entity_memory[entity_type].append({
                'value': entity_value,
                'confidence': confidence,
                'timestamp': datetime.now()
            })
            
            # 限制每个实体的记忆数量
            if len(self.entity_memory[entity_type]) > 5:
                self.entity_memory[entity_type] = self.entity_memory[entity_type][-5:]
                
        except Exception as e:
            logger.error(f"Entity addition failed: {e}")
    
    def get_entity(self, entity_type: str, default: str = None) -> Optional[str]:
        """获取实体值"""
        try:
            entities = self.entity_memory.get(entity_type, [])
            if entities:
                # 返回最近添加的实体
                return entities[-1]['value']
            return default
        except Exception:
            return default
    
    def set_preference(self, key: str, value: str) -> None:
        """设置用户偏好"""
        self.user_preferences[key] = value
        
    def get_preference(self, key: str, default: str = None) -> Optional[str]:
        """获取用户偏好"""
        return self.user_preferences.get(key, default)
    
    def add_session_goal(self, goal: str) -> None:
        """添加会话目标"""
        if goal not in self.session_goals:
            self.session_goals.append(goal)
            
    def mark_issue_resolved(self, issue: str) -> None:
        """标记问题已解决"""
        if issue not in self.resolved_issues:
            self.resolved_issues.append(issue)
            
    def is_issue_resolved(self, issue: str) -> bool:
        """检查问题是否已解决"""
        return issue in self.resolved_issues
    
    def _extract_topic(self, message: str) -> None:
        """提取对话主题（简化实现）"""
        # 关键词映射
        topic_keywords = {
            "订单": ["订单", "查询", "状态", "物流"],
            "产品": ["产品", "价格", "规格", "功能", "介绍"],
            "退款": ["退款", "退货", "换货", "取消"],
            "投诉": ["投诉", "不满", "问题", "糟糕", "差"],
            "技术支持": ["技术", "支持", "故障", "bug", "错误"]
        }
        
        message_lower = message.lower()
        for topic, keywords in topic_keywords.items():
            if any(keyword in message_lower for keyword in keywords):
                self.current_topic = topic
                break
        else:
            if not self.current_topic:
                self.current_topic = "一般咨询"
    
    def get_conversation_summary(self, last_n: int = 3) -> str:
        """获取对话摘要"""
        try:
            recent_messages = list(self.conversation_history)[-last_n:]
            summary_parts = []
            
            for msg in recent_messages:
                role_desc = "用户" if msg['user_id'] == self.user_id else "客服"
                summary_parts.append(f"{role_desc}: {msg['message'][:30]}...")
                
            return " | ".join(summary_parts)
            
        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            return ""
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'session_id': self.session_id,
            'user_id': self.user_id,
            'current_topic': self.current_topic,
            'emotion_state': self.emotion_state,
            'session_goals': self.session_goals,
            'resolved_issues': self.resolved_issues,
            'user_preferences': dict(self.user_preferences),
            'entity_memory': dict(self.entity_memory),
            'total_messages': self.total_messages,
            'last_update': self.last_update.isoformat(),
            'conversation_summary': self.get_conversation_summary()
        }
    
    def clear_context(self) -> None:
        """清空上下文"""
        self.conversation_history.clear()
        self.entity_memory.clear()
        self.session_goals.clear()
        self.resolved_issues.clear()
        self.current_topic = ""
        self.total_messages = 0
        
        logger.info(f"Context cleared for session: {self.session_id}")