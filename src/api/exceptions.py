from fastapi import HTTPException, status
from typing import Optional, Dict, Any

class BaseAPIException(HTTPException):
    """基础API异常类"""
    def __init__(
        self,
        status_code: int,
        detail: str,
        error_code: Optional[str] = None,
        headers: Optional[Dict[str, Any]] = None
    ):
        super().__init__(status_code=status_code, detail=detail, headers=headers)
        self.error_code = error_code

class AuthenticationError(BaseAPIException):
    """认证错误"""
    def __init__(self, detail: str = "Authentication failed"):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=detail,
            error_code="AUTH_001"
        )

class AuthorizationError(BaseAPIException):
    """授权错误"""
    def __init__(self, detail: str = "Permission denied"):
        super().__init__(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=detail,
            error_code="AUTH_002"
        )

class ValidationError(BaseAPIException):
    """数据验证错误"""
    def __init__(self, detail: str = "Validation failed"):
        super().__init__(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=detail,
            error_code="VAL_001"
        )

class NotFoundError(BaseAPIException):
    """资源未找到错误"""
    def __init__(self, resource: str = "Resource"):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"{resource} not found",
            error_code="RES_001"
        )

class ConflictError(BaseAPIException):
    """资源冲突错误"""
    def __init__(self, detail: str = "Resource conflict"):
        super().__init__(
            status_code=status.HTTP_409_CONFLICT,
            detail=detail,
            error_code="RES_002"
        )

class RateLimitError(BaseAPIException):
    """限流错误"""
    def __init__(self, detail: str = "Rate limit exceeded"):
        super().__init__(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=detail,
            error_code="RATE_001"
        )

class InternalServerError(BaseAPIException):
    """服务器内部错误"""
    def __init__(self, detail: str = "Internal server error"):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=detail,
            error_code="SRV_001"
        )

# 异常处理器
def register_exception_handlers(app):
    """注册异常处理器"""
    
    @app.exception_handler(AuthenticationError)
    async def auth_exception_handler(request, exc):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": exc.error_code,
                    "message": exc.detail,
                    "type": "authentication_error"
                }
            }
        )
    
    @app.exception_handler(ValidationError)
    async def validation_exception_handler(request, exc):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": exc.error_code,
                    "message": exc.detail,
                    "type": "validation_error"
                }
            }
        )
    
    @app.exception_handler(NotFoundError)
    async def not_found_exception_handler(request, exc):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": exc.error_code,
                    "message": exc.detail,
                    "type": "not_found_error"
                }
            }
        )
    
    @app.exception_handler(RateLimitError)
    async def rate_limit_exception_handler(request, exc):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": exc.error_code,
                    "message": exc.detail,
                    "type": "rate_limit_error"
                }
            },
            headers={"Retry-After": "60"}
        )