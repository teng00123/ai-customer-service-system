import time
import logging
from typing import Callable
from fastapi import FastAPI, Request, Response, HTTPException, status
from fastapi.middleware.base import BaseHTTPMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import RequestResponseEndpoint
from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta
from ..core.config import settings
from ..core.exceptions import AuthenticationError, RateLimitError

logger = logging.getLogger(__name__)

# 密码加密上下文
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class AuthenticationMiddleware(BaseHTTPMiddleware):
    """JWT认证中间件"""
    
    def __init__(self, app, exclude_paths: list = None):
        super().__init__(app)
        self.exclude_paths = exclude_paths or ["/", "/health", "/docs", "/redoc", "/openapi.json"]
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # 跳过排除路径
        if any(request.url.path.startswith(path) for path in self.exclude_paths):
            return await call_next(request)
        
        # 检查Authorization头
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing or invalid authorization header"
            )
        
        token = auth_header.split(" ")[1]
        try:
            payload = jwt.decode(token, settings.secret_key, algorithms=[settings.algorithm])
            request.state.user_id = payload.get("sub")
            request.state.username = payload.get("username")
        except JWTError as e:
            logger.warning(f"JWT decode error: {e}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired token"
            )
        
        return await call_next(request)

class RateLimitMiddleware(BaseHTTPMiddleware):
    """API限流中间件"""
    
    def __init__(self, app, calls: int = 100, period: int = 60):
        super().__init__(app)
        self.calls = calls
        self.period = period
        self.request_counts = {}
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        client_ip = request.client.host
        current_time = time.time()
        
        # 清理过期记录
        self.request_counts = {
            ip: [(t, count) for t, count in records 
                 if current_time - t < self.period]
            for ip, records in self.request_counts.items()
        }
        
        # 检查当前IP请求次数
        if client_ip in self.request_counts:
            total_requests = sum(count for _, count in self.request_counts[client_ip])
            if total_requests >= self.calls:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Rate limit exceeded"
                )
        
        response = await call_next(request)
        return response

class LoggingMiddleware(BaseHTTPMiddleware):
    """请求日志中间件"""
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        start_time = time.time()
        
        # 记录请求信息
        logger.info(f"Request: {request.method} {request.url} - Client: {request.client.host}")
        
        response = await call_next(request)
        
        # 记录响应信息
        process_time = time.time() - start_time
        logger.info(f"Response: {response.status_code} - Process time: {process_time:.4f}s")
        
        response.headers["X-Process-Time"] = str(process_time)
        return response