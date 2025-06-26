import os
from typing import List, Dict
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

class SecuritySettings:
    """Упрощённые настройки безопасности без Pydantic"""
    
    def __init__(self):
        # CORS настройки
        self.ALLOWED_ORIGINS = [
            "https://pikromachess-pttrns-frontend-dc0f.twc1.net"            
        ]
        
        # В development режиме добавляем localhost
        if os.getenv("ENVIRONMENT", "development") == "development":
            self.ALLOWED_ORIGINS.extend([
                "http://localhost:3000",
                "http://localhost:5173", 
                "http://127.0.0.1:3000",
                "http://127.0.0.1:5173"
            ])
        
        # Rate limiting
        self.RATE_LIMIT_ENABLED = True
        self.RATE_LIMIT_STORAGE_URL = "memory://"
        
        # Session settings
        self.SESSION_COOKIE_SECURE = os.getenv("ENVIRONMENT") != "development"
        self.SESSION_COOKIE_HTTPONLY = True
        self.SESSION_COOKIE_SAMESITE = "strict"
        
        # API настройки
        self.API_KEY_EXPIRY_HOURS = 1
        self.MAX_REQUEST_SIZE = 10 * 1024 * 1024  # 10MB
        
        # Validation настройки
        self.MAX_JSON_SIZE = 1024 * 1024  # 1MB
        self.MAX_STRING_LENGTH = 1000
        self.MAX_ARRAY_SIZE = 100
    
    # Headers безопасности (статическая константа)
    SECURITY_HEADERS = {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY", 
        "X-XSS-Protection": "1; mode=block",
        "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
        "Referrer-Policy": "strict-origin-when-cross-origin",
        "Permissions-Policy": "geolocation=(), microphone=(), camera=()"
    }

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        
        # Добавляем security headers
        for header, value in SecuritySettings.SECURITY_HEADERS.items():
            response.headers[header] = value
        
        # Удаляем headers, раскрывающие информацию о сервере
        response.headers.pop("server", None)
        response.headers.pop("x-powered-by", None)
        
        # Улучшенная CSP политика:
        csp_policy = (
            "default-src 'self'; "
            "script-src 'self'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: https:; "
            "connect-src 'self' https:; "
            "font-src 'self' data:; "
            "object-src 'none'; "
            "media-src 'self' blob:; "
            "frame-src 'none'; "
            "base-uri 'self'; "
            "form-action 'self'; "
            "upgrade-insecure-requests;"
        )
        response.headers["Content-Security-Policy"] = csp_policy
        
        return response

security_settings = SecuritySettings()