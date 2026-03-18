from fastapi import FastAPI, HTTPException, Depends, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer
import uvicorn
from typing import Dict, Any, List, Optional
import time
import logging
from datetime import datetime

# 导入本地模块
from .middleware import AuthenticationMiddleware, RateLimitMiddleware, LoggingMiddleware
from .exceptions import register_exception_handlers, BaseAPIException
from .models import *
from .auth import login, get_current_active_user, AuthService
from ..core.config import settings
from ..dialog_manager.manager import DialogManager
from ..intent_recognition.engine import IntentRecognizer
from ..knowledge_base.manager import KnowledgeBaseManager

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建FastAPI应用
app = FastAPI(
    title="AI智能客服系统",
    description="基于自然语言处理的智能客服API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# 注册异常处理器
register_exception_handlers(app)

# 添加中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(LoggingMiddleware)
app.add_middleware(RateLimitMiddleware, calls=100, period=60)
app.add_middleware(AuthenticationMiddleware)

# 初始化服务组件
dialog_manager = DialogManager()
intent_recognizer = IntentRecognizer()
kb_manager = KnowledgeBaseManager()

# ===== 基础API端点 =====

@app.get("/")
async def root() -> Dict[str, str]:
    """根路径"""
    return {"message": "AI智能客服系统API", "version": "1.0.0", "status": "running"}

@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """健康检查"""
    return {
        "status": "healthy", 
        "service": "ai-customer-service",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }

@app.get("/api/v1/info")
async def api_info() -> Dict[str, Any]:
    """API信息"""
    return {
        "name": "AI智能客服系统",
        "version": "1.0.0",
        "description": "基于自然语言处理的智能客服API",
        "endpoints": {
            "auth": ["/api/v1/auth/login", "/api/v1/auth/register"],
            "users": ["/api/v1/users/me", "/api/v1/users/profile"],
            "sessions": ["/api/v1/sessions", "/api/v1/sessions/{session_id}"],
            "chat": ["/api/v1/chat/message", "/api/v1/chat/response/{session_id}"]
        }
    }

# ===== 认证相关API =====

@app.post("/api/v1/auth/login", response_model=LoginResponse)
async def user_login(request: LoginRequest):
    """用户登录"""
    return await login(request)

@app.post("/api/v1/auth/logout")
async def user_logout(current_user: dict = Depends(get_current_active_user)):
    """用户登出"""
    # TODO: 实现token黑名单机制
    return {"message": "Logout successful"}

@app.post("/api/v1/auth/refresh")
async def refresh_token(current_user: dict = Depends(get_current_active_user)):
    """刷新访问令牌"""
    access_token_expires = timedelta(minutes=settings.access_token_expire_minutes)
    access_token = AuthService.create_access_token(
        data={"sub": current_user["username"], "user_id": current_user["id"]}, 
        expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

# ===== 用户管理API =====

@app.get("/api/v1/users/me", response_model=UserResponse)
async def get_current_user_info(current_user: dict = Depends(get_current_active_user)):
    """获取当前用户信息"""
    return UserResponse(**current_user)

@app.put("/api/v1/users/me", response_model=UserResponse)
async def update_current_user(
    user_update: UserUpdate,
    current_user: dict = Depends(get_current_active_user)
):
    """更新当前用户信息"""
    # TODO: 实现数据库更新逻辑
    updated_user = current_user.copy()
    if user_update.email:
        updated_user["email"] = user_update.email
    if user_update.full_name:
        updated_user["full_name"] = user_update.full_name
    if user_update.preferences:
        updated_user["preferences"].update(user_update.preferences)
    
    updated_user["updated_at"] = datetime.utcnow()
    return UserResponse(**updated_user)

@app.get("/api/v1/users/{user_id}", response_model=UserResponse)
async def get_user_by_id(
    user_id: str,
    current_user: dict = Depends(get_current_active_user)
):
    """根据ID获取用户信息"""
    # TODO: 实现数据库查询逻辑
    if user_id == current_user["id"] or current_user["role"] == "admin":
        # 模拟用户数据
        mock_user = current_user.copy()
        mock_user["id"] = user_id
        return UserResponse(**mock_user)
    else:
        raise HTTPException(status_code=403, detail="Permission denied")

# ===== 会话管理API =====

@app.post("/api/v1/sessions", response_model=SessionResponse)
async def create_session(
    session_create: SessionCreate,
    current_user: dict = Depends(get_current_active_user)
):
    """创建新会话"""
    if session_create.user_id != current_user["id"] and current_user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Permission denied")
    
    dialog = dialog_manager.create_dialog(session_create.user_id)
    
    return SessionResponse(
        id=dialog.session_id,
        session_id=dialog.session_id,
        user_id=dialog.user_id,
        status=dialog.status,
        context=dialog.context,
        created_at=dialog.created_at,
        updated_at=dialog.last_activity,
        last_activity=dialog.last_activity,
        message_count=len(dialog.conversation_history)
    )

@app.get("/api/v1/sessions/{session_id}", response_model=SessionResponse)
async def get_session(
    session_id: str,
    current_user: dict = Depends(get_current_active_user)
):
    """获取会话信息"""
    dialog = dialog_manager.get_dialog(session_id)
    if not dialog:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if dialog.user_id != current_user["id"] and current_user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Permission denied")
    
    return SessionResponse(
        id=dialog.session_id,
        session_id=dialog.session_id,
        user_id=dialog.user_id,
        status=dialog.status,
        context=dialog.context,
        created_at=dialog.created_at,
        updated_at=dialog.last_activity,
        last_activity=dialog.last_activity,
        message_count=len(dialog.conversation_history)
    )

@app.put("/api/v1/sessions/{session_id}", response_model=SessionResponse)
async def update_session(
    session_id: str,
    session_update: SessionUpdate,
    current_user: dict = Depends(get_current_active_user)
):
    """更新会话信息"""
    dialog = dialog_manager.get_dialog(session_id)
    if not dialog:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if dialog.user_id != current_user["id"] and current_user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Permission denied")
    
    if session_update.status:
        dialog.status = session_update.status
    if session_update.context:
        dialog.context.update(session_update.context)
    
    dialog.update_activity()
    
    return SessionResponse(
        id=dialog.session_id,
        session_id=dialog.session_id,
        user_id=dialog.user_id,
        status=dialog.status,
        context=dialog.context,
        created_at=dialog.created_at,
        updated_at=dialog.last_activity,
        last_activity=dialog.last_activity,
        message_count=len(dialog.conversation_history)
    )

@app.delete("/api/v1/sessions/{session_id}")
async def delete_session(
    session_id: str,
    current_user: dict = Depends(get_current_active_user)
):
    """关闭会话"""
    dialog = dialog_manager.get_dialog(session_id)
    if not dialog:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if dialog.user_id != current_user["id"] and current_user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Permission denied")
    
    success = dialog_manager.close_dialog(session_id)
    if success:
        return {"message": "Session closed successfully"}
    else:
        raise HTTPException(status_code=500, detail="Failed to close session")

@app.get("/api/v1/sessions", response_model=List[SessionResponse])
async def list_user_sessions(
    current_user: dict = Depends(get_current_active_user),
    limit: int = 20,
    offset: int = 0
):
    """获取用户会话列表"""
    # TODO: 从数据库获取会话列表
    # 这里简化处理，返回空列表
    return []

# ===== 消息处理API =====

@app.post("/api/v1/chat/message", response_model=ChatResponse)
async def send_message(
    chat_request: ChatRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_active_user)
):
    """发送聊天消息"""
    if chat_request.user_id != current_user["id"] and current_user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Permission denied")
    
    start_time = time.time()
    
    # 处理消息
    result = dialog_manager.process_message(
        session_id=chat_request.session_id or str(int(time.time())),
        user_id=chat_request.user_id,
        message=chat_request.message
    )
    
    processing_time = int((time.time() - start_time) * 1000)
    
    return ChatResponse(
        reply=result["reply"],
        intent=result["intent"],
        confidence=result["confidence"],
        entities=result["entities"],
        session_id=result["session_id"],
        suggested_actions=[],
        processing_time_ms=processing_time,
        model_version="v1.0.0"
    )

@app.get("/api/v1/chat/response/{session_id}")
async def get_chat_response(
    session_id: str,
    current_user: dict = Depends(get_current_active_user)
):
    """获取聊天回复"""
    dialog = dialog_manager.get_dialog(session_id)
    if not dialog:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if dialog.user_id != current_user["id"] and current_user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Permission denied")
    
    # 简化处理，返回最后一条助理消息
    assistant_messages = [msg for msg in dialog.conversation_history if msg["role"] == "assistant"]
    if assistant_messages:
        last_message = assistant_messages[-1]
        return {
            "reply": last_message["content"],
            "intent": last_message.get("intent"),
            "confidence": 0.88,
            "session_id": session_id
        }
    else:
        return {
            "reply": "欢迎使用AI客服系统！请问有什么可以帮助您的？",
            "intent": "greeting",
            "confidence": 0.95,
            "session_id": session_id
        }

@app.get("/api/v1/sessions/{session_id}/messages", response_model=List[MessageResponse])
async def get_session_messages(
    session_id: str,
    current_user: dict = Depends(get_current_active_user),
    limit: int = 50,
    offset: int = 0
):
    """获取会话消息历史"""
    dialog = dialog_manager.get_dialog(session_id)
    if not dialog:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if dialog.user_id != current_user["id"] and current_user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Permission denied")
    
    messages = []
    for i, msg in enumerate(dialog.conversation_history[offset:offset+limit]):
        messages.append(MessageResponse(
            id=f"{session_id}_{i}",
            session_id=session_id,
            role=msg["role"],
            content=msg["content"],
            intent=msg.get("intent"),
            confidence_score=None,
            entities=None,
            sentiment_score=None,
            created_at=datetime.fromisoformat(msg["timestamp"]) if "timestamp" in msg else datetime.utcnow()
        ))
    
    return messages

# ===== 知识库API =====

@app.post("/api/v1/knowledge/search")
async def search_knowledge(
    query: str,
    size: int = 5,
    current_user: dict = Depends(get_current_active_user)
):
    """搜索知识库"""
    results = kb_manager.search(query, size=size)
    return {"query": query, "results": results, "total": len(results)}

@app.post("/api/v1/knowledge/documents")
async def add_knowledge_document(
    question: str,
    answer: str,
    category: str = "general",
    tags: List[str] = None,
    current_user: dict = Depends(get_current_active_user)
):
    """添加知识文档"""
    if current_user["role"] not in ["admin", "service_agent"]:
        raise HTTPException(status_code=403, detail="Permission denied")
    
    success = kb_manager.add_document(question, answer, category, tags)
    if success:
        return {"message": "Document added successfully"}
    else:
        raise HTTPException(status_code=500, detail="Failed to add document")

if __name__ == "__main__":
    uvicorn.run(
        "main:app", 
        host=settings.host, 
        port=settings.port, 
        reload=settings.debug
    )