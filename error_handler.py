import uuid
from datetime import datetime
from typing import Any, Dict

class SecureErrorHandler:
    
    ERROR_CODES = {
        "VALIDATION_ERROR": "Input validation failed",
        "AUTHENTICATION_ERROR": "Authentication required",
        "AUTHORIZATION_ERROR": "Access denied",
        "RATE_LIMIT_ERROR": "Rate limit exceeded",
        "NETWORK_ERROR": "Network connectivity issue",
        "SERVER_ERROR": "Internal server error",
        "UNKNOWN_ERROR": "An unexpected error occurred"
    }
    
    @staticmethod
    def handle_error(error: Any, context: str = "") -> Dict[str, Any]:
        """Безопасная обработка ошибок"""
        request_id = str(uuid.uuid4())[:8]
        timestamp = datetime.utcnow().isoformat()
        
        # Определяем тип ошибки
        error_code = SecureErrorHandler._classify_error(error)
        safe_message = SecureErrorHandler.ERROR_CODES.get(error_code, "An error occurred")
        
        # Логируем детальную ошибку (для разработчиков)
        error_details = {
            "request_id": request_id,
            "context": context,
            "error_type": type(error).__name__ if error else "Unknown",
            "error_message": str(error)[:200] if error else "No error object",  # Ограничиваем длину
            "timestamp": timestamp
        }
        
        print(f"[ERROR_HANDLER] {error_details}")
        
        # Возвращаем безопасный ответ
        return {
            "code": error_code,
            "message": safe_message,
            "request_id": request_id,
            "timestamp": timestamp
        }
    
    @staticmethod
    def _classify_error(error: Any) -> str:
        """Классификация ошибки"""
        if not error:
            return "UNKNOWN_ERROR"
        
        error_str = str(error).lower()
        
        if any(keyword in error_str for keyword in ["validation", "invalid", "format"]):
            return "VALIDATION_ERROR"
        
        if any(keyword in error_str for keyword in ["unauthorized", "authentication", "login"]):
            return "AUTHENTICATION_ERROR"
        
        if any(keyword in error_str for keyword in ["forbidden", "access", "permission"]):
            return "AUTHORIZATION_ERROR"
        
        if any(keyword in error_str for keyword in ["rate limit", "too many", "quota"]):
            return "RATE_LIMIT_ERROR"
        
        if any(keyword in error_str for keyword in ["network", "connection", "timeout"]):
            return "NETWORK_ERROR"
        
        if hasattr(error, "status_code") and getattr(error, "status_code") >= 500:
            return "SERVER_ERROR"
        
        return "UNKNOWN_ERROR"