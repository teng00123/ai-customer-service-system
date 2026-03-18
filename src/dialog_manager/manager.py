import logging
import uuid
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from ..core.config import settings
from ..knowledge_base.manager import KnowledgeBaseManager
from ..intent_recognition.engine import IntentRecognizer

logger = logging.getLogger(__name__)

class DialogState:
    """对话状态"""
    def __init__(self, session_id: str, user_id: str):
        self.session_id = session_id
        self.user_id = user_id
        self.context = {}
        self.current_intent = None
        self.conversation_history = []
        self.created_at = datetime.now()
        self.last_activity = datetime.now()
        self.status = "active"  # active, closed, transferred
        
    def update_activity(self):
        self.last_activity = datetime.now()
        
    def add_message(self, role: str, content: str, intent: str = None):
        self.conversation_history.append({
            "role": role,
            "content": content,
            "intent": intent,
            "timestamp": datetime.now().isoformat()
        })
        self.update_activity()
        
    def get_context_value(self, key: str, default=None):
        return self.context.get(key, default)
        
    def set_context_value(self, key: str, value: Any):
        self.context[key] = value
        self.update_activity()

class DialogManager:
    """对话管理器"""
    
    def __init__(self, redis_uri: str = None):
        self.redis_uri = redis_uri or settings.redis_uri
        self.active_dialogs = {}  # 内存存储，生产环境应使用Redis
        self.kb_manager = KnowledgeBaseManager()
        self.intent_recognizer = IntentRecognizer()
        
    def create_dialog(self, user_id: str) -> DialogState:
        """创建新的对话"""
        session_id = str(uuid.uuid4())
        dialog_state = DialogState(session_id, user_id)
        self.active_dialogs[session_id] = dialog_state
        logger.info(f"Created dialog: {session_id} for user: {user_id}")
        return dialog_state
        
    def get_dialog(self, session_id: str) -> Optional[DialogState]:
        """获取对话状态"""
        dialog = self.active_dialogs.get(session_id)
        
        # 检查超时（30分钟无活动）
        if dialog and datetime.now() - dialog.last_activity > timedelta(minutes=30):
            dialog.status = "closed"
            del self.active_dialogs[session_id]
            return None
            
        return dialog
        
    def process_message(self, session_id: str, user_id: str, message: str) -> Dict[str, Any]:
        """处理用户消息"""
        # 获取或创建对话
        dialog = self.get_dialog(session_id)
        if not dialog:
            dialog = self.create_dialog(user_id)
            
        # 记录用户消息
        dialog.add_message("user", message)
        
        # 意图识别
        intent, confidence = self.intent_recognizer.recognize_intent(message)
        entities = self.intent_recognizer.extract_entities(message)
        
        # 记录助理消息（临时）
        dialog.current_intent = intent
        dialog.set_context_value("last_intent", intent)
        dialog.set_context_value("last_confidence", confidence)
        dialog.set_context_value("entities", entities)
        
        # 知识库搜索
        kb_results = self.kb_manager.search(message, size=3)
        
        # 生成回复（简化版）
        if intent == "greeting":
            reply = "您好！我是AI客服助手，很高兴为您服务。请问有什么可以帮助您的？"
        elif intent == "goodbye":
            reply = "感谢您的使用，再见！"
            dialog.status = "closed"
        elif kb_results:
            best_result = kb_results[0]
            reply = best_result["answer"]
        else:
            reply = "抱歉，我没有理解您的问题。请您换个方式描述，或者联系人工客服。"
            
        # 记录助理回复
        dialog.add_message("assistant", reply, intent)
        
        return {
            "reply": reply,
            "intent": intent,
            "confidence": confidence,
            "entities": entities,
            "session_id": session_id,
            "kb_results": kb_results
        }
        
    def close_dialog(self, session_id: str) -> bool:
        """关闭对话"""
        if session_id in self.active_dialogs:
            self.active_dialogs[session_id].status = "closed"
            del self.active_dialogs[session_id]
            return True
        return False