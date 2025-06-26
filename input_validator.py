import re
import html
from typing import Dict, Any, Optional
import os


class InputValidator:
    
    @staticmethod
    def validate_api_key(api_key: str) -> bool:
        """Валидация формата API ключа"""
        if not isinstance(api_key, str):
            return False
        
        # API ключ должен быть hex строкой длиной 64 символа
        if len(api_key) != 64:
            return False
        
        # Только hex символы
        if not re.match(r'^[a-fA-F0-9]{64}$', api_key):
            return False
        
        return True
    
    @staticmethod
    def validate_nft_metadata(metadata: Any) -> Dict[str, Any]:
        """Валидация и санитизация метаданных NFT"""
        if not isinstance(metadata, dict):
            return {
                "is_valid": False,
                "error": "Metadata must be a dictionary",
                "sanitized": {}
            }
        
        sanitized = {}
        
        # Валидация обязательных полей
        name = metadata.get("name")
        if not name or not isinstance(name, str):
            return {
                "is_valid": False,
                "error": "Name is required and must be a string",
                "sanitized": {}
            }
        
        # Санитизация name
        sanitized_name = InputValidator._sanitize_string(name, 200)
        if not sanitized_name:
            return {
                "is_valid": False,
                "error": "Invalid name content",
                "sanitized": {}
            }
        
        sanitized["name"] = sanitized_name
        
        # Валидация опциональных полей
        optional_fields = {
            "description": 1000,
            "image": 500,
            "animation_url": 500
        }
        
        for field, max_length in optional_fields.items():
            value = metadata.get(field)
            if value is not None:
                if not isinstance(value, str):
                    return {
                        "is_valid": False,
                        "error": f"{field} must be a string",
                        "sanitized": {}
                    }
                
                if field in ["image", "animation_url"]:
                    if not InputValidator._validate_url(value):
                        return {
                            "is_valid": False,
                            "error": f"Invalid {field} URL",
                            "sanitized": {}
                        }
                
                sanitized_value = InputValidator._sanitize_string(value, max_length)
                if sanitized_value:
                    sanitized[field] = sanitized_value
        
        # Валидация атрибутов
        attributes = metadata.get("attributes")
        if attributes is not None:
            if not isinstance(attributes, list):
                return {
                    "is_valid": False,
                    "error": "Attributes must be an array",
                    "sanitized": {}
                }
            
            if len(attributes) > 50:
                return {
                    "is_valid": False,
                    "error": "Too many attributes (max 50)",
                    "sanitized": {}
                }
            
            sanitized_attributes = []
            for attr in attributes[:50]:  # Ограничиваем количество
                if not isinstance(attr, dict):
                    continue
                
                sanitized_attr = {}
                
                trait_type = attr.get("trait_type")
                if trait_type and isinstance(trait_type, str):
                    sanitized_trait = InputValidator._sanitize_string(trait_type, 100)
                    if sanitized_trait:
                        sanitized_attr["trait_type"] = sanitized_trait
                
                value = attr.get("value")
                if value is not None:
                    if isinstance(value, str):
                        sanitized_value = InputValidator._sanitize_string(value, 200)
                        if sanitized_value:
                            sanitized_attr["value"] = sanitized_value
                    elif isinstance(value, (int, float)) and -1000000 <= value <= 1000000:
                        sanitized_attr["value"] = value
                
                if sanitized_attr:
                    sanitized_attributes.append(sanitized_attr)
            
            if sanitized_attributes:
                sanitized["attributes"] = sanitized_attributes
        
        return {
            "is_valid": True,
            "error": None,
            "sanitized": sanitized
        }
    
    @staticmethod
    def _sanitize_string(value: str, max_length: int) -> Optional[str]:
        """Санитизация строки"""
        if not isinstance(value, str):
            return None
        
        # Удаляем нулевые байты и контрольные символы
        sanitized = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', value)
        
        # Экранируем HTML
        sanitized = html.escape(sanitized)
        
        # Обрезаем до максимальной длины
        sanitized = sanitized[:max_length]
        
        # Проверяем на потенциально опасный контент
        dangerous_patterns = [
            r'<script.*?>.*?</script>',
            r'javascript:',
            r'vbscript:',
            r'data:text/html',
            r'on\w+\s*=',
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, sanitized, re.IGNORECASE):
                return None
        
        return sanitized.strip()
    
    @staticmethod
    def _validate_url(url: str) -> bool:
        """Валидация URL"""
        if not isinstance(url, str) or len(url) > 500:
            return False
        
        # Простая проверка формата URL
        url_pattern = r'^https://[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,}(/[^\s]*)?$'
        if not re.match(url_pattern, url):
            return False
        
        # Проверяем на подозрительные домены
        suspicious_domains = [
            'localhost',
            '127.0.0.1',
            '0.0.0.0',
            '10.',
            '192.168.',
            '172.16.'
        ]
        
        for domain in suspicious_domains:
            if domain in url.lower():
                return False
        
        return True
    
    @staticmethod
    def validate_ip_address(ip: str) -> bool:
        """Валидация IP адреса"""
        if not isinstance(ip, str):
            return False
        
        # IPv4 валидация
        try:
            parts = ip.split('.')
            if len(parts) != 4:
                return False
            
            for part in parts:
                if not part.isdigit():
                    return False
                num = int(part)
                if num < 0 or num > 255:
                    return False
            
            # Проверяем на частные/зарезервированные адреса
            private_ranges = [
                ('10.0.0.0', '10.255.255.255'),
                ('172.16.0.0', '172.31.255.255'),
                ('192.168.0.0', '192.168.255.255'),
                ('127.0.0.0', '127.255.255.255'),
                ('0.0.0.0', '0.255.255.255')
            ]
            
            # Для безопасности отвергаем частные адреса в продакшене
            if os.getenv("ENVIRONMENT") == "production":
                ip_int = sum(int(part) << (8 * (3 - i)) for i, part in enumerate(parts))
                for start, end in private_ranges:
                    start_int = sum(int(part) << (8 * (3 - i)) for i, part in enumerate(start.split('.')))
                    end_int = sum(int(part) << (8 * (3 - i)) for i, part in enumerate(end.split('.')))
                    if start_int <= ip_int <= end_int:
                        return False
            
            return True
            
        except (ValueError, IndexError):
            return False