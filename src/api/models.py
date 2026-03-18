from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

class UserRole(str, Enum):
    """用户角色枚举"""
    ADMIN = "admin"
    USER = "user"
    SERVICE_AGENT = "service_agent"

class UserBase(BaseModel):
    """用户基础模型"""
    username: str = Field(..., min_length=3, max_length=50, regex="^[a-zA-Z0-9_]+$")
    email: str = Field(..., regex="^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\\.[a-zA-Z0-9-.]+$")
    full_name: Optional[str] = Field(None, max_length=100)
    role: UserRole = Field(default=UserRole.USER)
    preferences: Dict[str, Any] = Field(default_factory=dict)

class UserCreate(UserBase):
    """用户创建模型"""
    password: str = Field(..., min_length=8, max_length=128)

class UserUpdate(BaseModel):
    """用户更新模型"""
    email: Optional[str] = Field(None, regex="^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\\.[a-zA-Z0-9-.]+$")
    full_name: Optional[str] = Field(None, max_length=100)
    preferences: Optional[Dict[str, Any]] = None

class UserResponse(UserBase):
    """用户响应模型"""
    id: str
    created_at: datetime
    updated_at: datetime
    is_active: bool = True
    
    class Config:
        from_attributes = True

class LoginRequest(BaseModel):
    """登录请求模型"""
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=8, max_length=128)

class LoginResponse(BaseModel):
    """登录响应模型"""
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserResponse

class SessionBase(BaseModel):
    """会话基础模型"""
    user_id: str
    status: str = Field(default="active", regex="^(active|closed|transferred)$")
    context: Dict[str, Any] = Field(default_factory=dict)

class SessionCreate(SessionBase):
    """会话创建模型"""
    pass

class SessionUpdate(BaseModel):
    """会话更新模型"""
    status: Optional[str] = Field(None, regex="^(active|closed|transferred)$")
    context: Optional[Dict[str, Any]] = None

class SessionResponse(SessionBase):
    """会话响应模型"""
    id: str
    session_id: str
    created_at: datetime
    updated_at: datetime
    last_activity: datetime
    message_count: int = 0
    
    class Config:
        from_attributes = True

class MessageBase(BaseModel):
    """消息基础模型"""
    session_id: str
    role: str = Field(..., regex="^(user|assistant|system)$") 
    content: str = Field(..., min_length=1, max_length=5000)
    intent: Optional[str] = Field(None, max_length=100)
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    entities: Optional[Dict[str, List[str]]] = None
    sentiment_score: Optional[float] = Field(None, ge=-1.0, le=1.0)

class MessageCreate(MessageBase):
    """消息创建模型"""
    pass

class MessageResponse(MessageBase):
    """消息响应模型"""
    id: str
    created_at: datetime
    processing_time_ms: Optional[int] = None
    model_version: Optional[str] = None
    
    class Config:
        from_attributes = True

class ChatRequest(BaseModel):
    """聊天请求模型"""
    session_id: Optional[str] = None
    user_id: str
    message: str = Field(..., min_length=1, max_length=5000)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ChatResponse(BaseModel):
    """聊天响应模型"""
    reply: str
    intent: Optional[str] = None
    confidence: Optional[float] = None
    entities: Optional[Dict[str, List[str]]] = None
    session_id: str
    suggested_actions: List[Dict[str, Any]] = Field(default_factory=list)
    processing_time_ms: int
    model_version: str