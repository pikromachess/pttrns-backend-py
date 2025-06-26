from typing import Dict, Optional
import time
import asyncio
from collections import defaultdict, deque
from fastapi import HTTPException, Request
import hashlib

class AdvancedRateLimiter:
    def __init__(self):
        self.requests: Dict[str, deque] = defaultdict(deque)
        self.blocked_ips: Dict[str, float] = {}
        self.suspicious_activities: Dict[str, int] = defaultdict(int)
    
    def get_client_id(self, request: Request) -> str:
        # Получаем IP адрес с учетом прокси
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            client_ip = forwarded_for.split(",")[0].strip()
        else:
            client_ip = request.client.host if request.client else "unknown"
        
        # Хешируем IP для privacy
        return hashlib.sha256(client_ip.encode()).hexdigest()[:16]
    
    def is_blocked(self, client_id: str) -> bool:
        if client_id in self.blocked_ips:
            if time.time() < self.blocked_ips[client_id]:
                return True
            else:
                del self.blocked_ips[client_id]
        return False
    
    def block_client(self, client_id: str, duration: int = 300):  # 5 минут
        self.blocked_ips[client_id] = time.time() + duration
        print(f"Blocked client {client_id} for {duration} seconds")
    
    def check_rate_limit(self, request: Request, max_requests: int, window_seconds: int) -> bool:
        client_id = self.get_client_id(request)
        
        # Проверяем блокировку
        if self.is_blocked(client_id):
            raise HTTPException(status_code=429, detail="IP temporarily blocked due to rate limiting")
        
        current_time = time.time()
        client_requests = self.requests[client_id]
        
        # Удаляем старые запросы
        while client_requests and client_requests[0] <= current_time - window_seconds:
            client_requests.popleft()
        
        # Проверяем лимит
        if len(client_requests) >= max_requests:
            # Увеличиваем счетчик подозрительной активности
            self.suspicious_activities[client_id] += 1
            
            # Блокируем при превышении лимита несколько раз
            if self.suspicious_activities[client_id] >= 3:
                self.block_client(client_id, 600)  # 10 минут
            elif self.suspicious_activities[client_id] >= 2:
                self.block_client(client_id, 300)  # 5 минут
            
            raise HTTPException(
                status_code=429, 
                detail=f"Rate limit exceeded: {max_requests} requests per {window_seconds} seconds"
            )
        
        # Добавляем текущий запрос
        client_requests.append(current_time)
        
        # Сбрасываем подозрительную активность при нормальном использовании
        if len(client_requests) <= max_requests // 2:
            self.suspicious_activities[client_id] = max(0, self.suspicious_activities[client_id] - 1)
        
        return True
    
    def cleanup_old_records(self):
        """Очистка старых записей"""
        current_time = time.time()
        
        # Очищаем старые запросы
        for client_id in list(self.requests.keys()):
            client_requests = self.requests[client_id]
            while client_requests and client_requests[0] <= current_time - 3600:  # 1 час
                client_requests.popleft()
            
            if not client_requests:
                del self.requests[client_id]
        
        # Очищаем старые блокировки
        for client_id in list(self.blocked_ips.keys()):
            if current_time >= self.blocked_ips[client_id]:
                del self.blocked_ips[client_id]

rate_limiter = AdvancedRateLimiter()

# Функция для использования в эндпоинтах
def create_rate_limit_dependency(max_requests: int, window_seconds: int):
    def rate_limit_dependency(request: Request):
        return rate_limiter.check_rate_limit(request, max_requests, window_seconds)
    return rate_limit_dependency

# Периодическая очистка
async def cleanup_task():
    while True:
        await asyncio.sleep(300)  # Каждые 5 минут
        rate_limiter.cleanup_old_records()