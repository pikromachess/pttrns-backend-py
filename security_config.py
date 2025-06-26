import os
from typing import List
from pydantic import BaseSettings
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

class SecuritySettings(BaseSettings):
    # CORS настройки
    ALLOWED_ORIGINS: List[str] = [
        "https://your-domain.com",
        "https://www.your-domain.com"
    ]
    
    # В development режиме добавляем localhost
    if os.getenv("ENVIRONMENT") == "development":
        ALLOWED_ORIGINS.extend([
            "http://localhost:3000",
            "http://localhost:5173",
            "http://127.0.0.1:3000",
            "http://127.0.0.1:5173"
        ])
    
    # Headers безопасности
    SECURITY_HEADERS = {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
        "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
        "Referrer-Policy": "strict-origin-when-cross-origin",
        "Permissions-Policy": "geolocation=(), microphone=(), camera=()"
    }
    
    # Rate limiting
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_STORAGE_URL: str = "memory://"
    
    # Session settings
    SESSION_COOKIE_SECURE: bool = True
    SESSION_COOKIE_HTTPONLY: bool = True
    SESSION_COOKIE_SAMESITE: str = "strict"
    
    # API настройки
    API_KEY_EXPIRY_HOURS: int = 1
    MAX_REQUEST_SIZE: int = 10 * 1024 * 1024  # 10MB
    
    # Validation настройки
    MAX_JSON_SIZE: int = 1024 * 1024  # 1MB
    MAX_STRING_LENGTH: int = 1000
    MAX_ARRAY_SIZE: int = 100
    
    class Config:
        env_file = ".env"

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        
        # Добавляем security headers
        for header, value in security_settings.SECURITY_HEADERS.items():
            response.headers[header] = value
        
        # Удаляем headers, раскрывающие информацию о сервере
        response.headers.pop("server", None)
        response.headers.pop("x-powered-by", None)
        
        # Улучшенная CSP политика:
        csp_policy = (
            "default-src 'self'; "
            "script-src 'self'; "  # Убираем 'unsafe-inline'
            "style-src 'self' 'unsafe-inline'; "  # Оставляем только для стилей
            "img-src 'self' data: https:; "
            "connect-src 'self' https:; "  # Разрешаем HTTPS соединения
            "font-src 'self' data:; "
            "object-src 'none'; "
            "media-src 'self' blob:; "  # Для аудио потоков
            "frame-src 'none'; "
            "base-uri 'self'; "
            "form-action 'self'; "
            "upgrade-insecure-requests;"
        )
        response.headers["Content-Security-Policy"] = csp_policy
        
        return response

security_settings = SecuritySettings()