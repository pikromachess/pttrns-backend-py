import asyncio
import json
import hashlib
from datetime import datetime
from typing import Any, Dict, Optional
import re

class SecureLogger:
    
    @staticmethod
    async def _hash_sensitive_data(data: str) -> str:
        """Хеширование чувствительных данных"""
        if len(data) < 8:
            return "[REDACTED]"
        
        hash_obj = hashlib.sha256(data.encode())
        hash_hex = hash_obj.hexdigest()
        
        return f"{data[:3]}...{data[-3:]} (hash: {hash_hex[:8]})"
    
    @staticmethod
    async def _sanitize_data(data: Any) -> Any:
        """Санитизация данных для логирования"""
        if isinstance(data, str):
            # Потенциально чувствительные данные
            if len(data) > 20 and re.match(r'^[A-Za-z0-9+/=_-]+$', data):
                return await SecureLogger._hash_sensitive_data(data)
            return data
        
        elif isinstance(data, dict):
            sanitized = {}
            sensitive_keys = ['password', 'secret', 'token', 'key', 'auth', 'api']
            
            for key, value in data.items():
                if any(sensitive in key.lower() for sensitive in sensitive_keys):
                    if isinstance(value, str) and len(value) > 5:
                        sanitized[key] = await SecureLogger._hash_sensitive_data(value)
                    else:
                        sanitized[key] = "[REDACTED]"
                else:
                    sanitized[key] = await SecureLogger._sanitize_data(value)
            
            return sanitized
        
        elif isinstance(data, list):
            return [await SecureLogger._sanitize_data(item) for item in data[:10]]  # Ограничиваем массивы
        
        return data
    
    @staticmethod
    async def info(message: str, context: Optional[Dict] = None):
        """Информационное логирование"""
        sanitized_context = await SecureLogger._sanitize_data(context) if context else {}
        
        log_entry = {
            "level": "INFO",
            "message": message,
            "context": sanitized_context,
            "timestamp": datetime.utcnow().isoformat(),
        }
        
        print(f"[INFO] {json.dumps(log_entry, ensure_ascii=False)}")
    
    @staticmethod
    async def error(message: str, error: Optional[Exception] = None, context: Optional[Dict] = None):
        """Логирование ошибок"""
        sanitized_context = await SecureLogger._sanitize_data(context) if context else {}
        
        error_info = None
        if error:
            error_info = {
                "type": type(error).__name__,
                "message": str(error)[:200],  # Ограничиваем длину сообщения
            }
        
        log_entry = {
            "level": "ERROR",
            "message": message,
            "error": error_info,
            "context": sanitized_context,
            "timestamp": datetime.utcnow().isoformat(),
        }
        
        print(f"[ERROR] {json.dumps(log_entry, ensure_ascii=False)}")
    
    @staticmethod
    async def security(event: str, details: Dict):
        """Логирование событий безопасности"""
        sanitized_details = await SecureLogger._sanitize_data(details)
        
        log_entry = {
            "level": "SECURITY",
            "event": event,
            "details": sanitized_details,
            "timestamp": datetime.utcnow().isoformat(),
            "severity": "warning"
        }
        
        print(f"[SECURITY] {json.dumps(log_entry, ensure_ascii=False)}")
    
    @staticmethod
    async def user_action(action: str, user_id: str, details: Optional[Dict] = None):
        """Логирование действий пользователей"""
        hashed_user_id = await SecureLogger._hash_sensitive_data(user_id)
        sanitized_details = await SecureLogger._sanitize_data(details) if details else {}
        
        log_entry = {
            "level": "USER_ACTION",
            "action": action,
            "user_id": hashed_user_id,
            "details": sanitized_details,
            "timestamp": datetime.utcnow().isoformat(),
        }
        
        print(f"[USER] {json.dumps(log_entry, ensure_ascii=False)}")