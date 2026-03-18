"""对话会话模型 - Claude Code风格"""
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

class MessageType(Enum):
    """消息类型枚举"""
    TEXT = "text"
    IMAGE = "image"
    FILE = "file"
    VOICE = "voice"

@dataclass
class DialogMessage:
    """对话消息模型"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    role: str = "user"  # user, assistant, system
    content: str = ""
    message_type: MessageType = MessageType.TEXT
    timestamp: datetime = field(default_factory=datetime.now)
    intent: Dict[str, Any] = field(default_factory=dict)
    entities: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'role': self.role,
            'content': self.content,
            'message_type': self.message_type.value,
            'timestamp': self.timestamp.isoformat(),
            'intent': self.intent,
            'entities': self.entities,
            'metadata': self.metadata
        }

@dataclass
class DialogSession:
    """对话会话模型"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    status: str = "active"  # active, closed, transferred
    messages: List[DialogMessage] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    context_summary: str = ""
    satisfaction_score: Optional[float] = None
    
    def add_message(self, role: str, content: str, intent: Dict[str, Any] = None,
                   entities: Dict[str, Any] = None, message_type: MessageType = MessageType.TEXT,
                   metadata: Dict[str, Any] = None) -> DialogMessage:
        """添加消息到会话"""
        message = DialogMessage(
            role=role,
            content=content,
            message_type=message_type,
            intent=intent or {},
            entities=entities or {},
            metadata=metadata or {}
        )
        self.messages.append(message)
        self.last_activity = datetime.now()
        return message
    
    def update_activity(self) -> None:
        """更新会话活动时间"""
        self.last_activity = datetime.now()
    
    def close(self) -> None:
        """关闭会话"""
        self.status = "closed"
        self.last_activity = datetime.now()
    
    def transfer_to_human(self) -> None:
        """转接人工客服"""
        self.status = "transferred"
        self.last_activity = datetime.now()
    
    def get_recent_messages(self, count: int = 5) -> List[DialogMessage]:
        """获取最近的消息"""
        return self.messages[-count:] if len(self.messages) > count else self.messages[:]
    
    def get_messages_by_role(self, role: str) -> List[DialogMessage]:
        """根据角色获取消息"""
        return [msg for msg in self.messages if msg.role == role]
    
    def summarize_context(self) -> str:
        """生成上下文摘要"""
        if self.context_summary:
            return self.context_summary
            
        recent_messages = self.get_recent_messages(3)
        summary_parts = []
        
        for msg in recent_messages:
            if msg.role == "user" and msg.content:
                summary_parts.append(f"用户: {msg.content[:50]}...")
            elif msg.role == "assistant" and msg.content:
                summary_parts.append(f"客服: {msg.content[:50]}...")
                
        self.context_summary = " | ".join(summary_parts)
        return self.context_summary
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'user_id': self.user_id,
            'created_at': self.created_at.isoformat(),
            'last_activity': self.last_activity.isoformat(),
            'status': self.status,
            'message_count': len(self.messages),
            'messages': [msg.to_dict() for msg in self.messages],
            'metadata': self.metadata,
            'context_summary': self.context_summary,
            'satisfaction_score': self.satisfaction_score
        }