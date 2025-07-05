import os
import sys
import logging
import jwt
import time
import hashlib
from fastapi import FastAPI, HTTPException, Request, Depends, Header
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import uvicorn
import asyncio
import aiohttp
from contextlib import asynccontextmanager
from typing import Dict, Any, Iterator, Optional
from io import BytesIO
import json
import secrets
import hashlib
import time

# Настройка логирования для Timeweb Cloud
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Условные импорты с проверкой наличия модулей
def safe_import():
    try:
        from database import db_manager
        logger.info("✅ Database модуль импортирован")
        return db_manager
    except ImportError as e:
        logger.warning(f"⚠️ Database модуль недоступен: {e}")
        return None

def safe_import_audio():
    try:
        from audio_downloader import audio_downloader
        from music_generator import streaming_music_generator
        logger.info("✅ Audio модули импортированы")
        return audio_downloader, streaming_music_generator
    except ImportError as e:
        logger.warning(f"⚠️ Audio модули недоступны: {e}")
        return None, None

def safe_import_config():
    try:
        from config import server_config
        logger.info("✅ Config модуль импортирован")
        return server_config
    except ImportError as e:
        logger.warning(f"⚠️ Config модуль недоступен: {e}")
        # Fallback конфигурация
        class FallbackConfig:
            host = os.getenv("HOST", "0.0.0.0")
            port = int(os.getenv("PORT", "8000"))
            debug = os.getenv("DEBUG", "false").lower() == "true"
        return FallbackConfig()

def safe_import_validator():
    try:
        from input_validator import InputValidator
        logger.info("✅ Validator модуль импортирован")
        return InputValidator()
    except ImportError as e:
        logger.warning(f"⚠️ Validator модуль недоступен: {e}")
        return None

# Безопасные импорты
db_manager = safe_import()
audio_downloader, streaming_music_generator = safe_import_audio()
server_config = safe_import_config()
validator = safe_import_validator()

# Базовые настройки
NODE_SERVER_URL = os.getenv("NODE_SERVER_URL", "http://localhost:3000")
BACKEND_SECRET = os.getenv("BACKEND_SECRET", "MY_SECRET_FROM_ENV")

# Настройки аутентификации
security = HTTPBearer()

# Упрощенная конфигурация CORS
ALLOWED_ORIGINS = [
    "https://pikromachess-pttrns-frontend-dc0f.twc1.net"
]

if os.getenv("ENVIRONMENT", "development") == "development":
    ALLOWED_ORIGINS.extend([
        "http://localhost:3000",
        "http://localhost:5173"
    ])

# Упрощенный rate limiter для сессий
class SessionRateLimiter:
    def __init__(self):
        self.requests = {}
    
    def check_session_limit(self, user_address: str, max_requests: int = 10, window: int = 60) -> bool:
        current_time = time.time()
        
        if user_address not in self.requests:
            self.requests[user_address] = []
        
        # Удаляем старые запросы
        self.requests[user_address] = [
            req_time for req_time in self.requests[user_address] 
            if current_time - req_time < window
        ]
        
        if len(self.requests[user_address]) >= max_requests:
            return False
        
        self.requests[user_address].append(current_time)
        return True

session_rate_limiter = SessionRateLimiter()

# Проверка сессионного токена
async def verify_session_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    
    try:
        # Декодируем JWT токен
        payload = jwt.decode(token, BACKEND_SECRET, algorithms=["HS256"])
        
        # Проверяем тип токена
        if payload.get("type") != "listening_session":
            raise HTTPException(status_code=401, detail="Неверный тип токена")
        
        # Проверяем срок действия (если не указан exp, проверяем timestamp + 1 час)
        current_time = int(time.time())
        token_timestamp = payload.get("timestamp", 0)
        
        # Токен действует 1 час
        if current_time - token_timestamp > 3600:
            raise HTTPException(status_code=401, detail="Сессия истекла")
        
        user_address = payload.get("address")
        if not user_address:
            raise HTTPException(status_code=401, detail="Некорректный токен сессии")
        
        # Проверяем rate limiting для пользователя
        if not session_rate_limiter.check_session_limit(user_address):
            raise HTTPException(status_code=429, detail="Превышен лимит запросов для пользователя")
        
        return {
            "address": user_address,
            "domain": payload.get("domain"),
            "timestamp": token_timestamp,
            "valid": True
        }
        
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Сессия истекла")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Недействительный токен сессии")
    except Exception as e:
        logger.error(f"Ошибка проверки токена: {e}")
        raise HTTPException(status_code=401, detail="Ошибка проверки токена")

# Упрощенная валидация запроса
def validate_music_request(data: dict) -> dict:
    """Упрощенная валидация музыкального запроса"""
    if not isinstance(data, dict):
        raise HTTPException(status_code=400, detail="Request must be a JSON object")
    
    metadata = data.get("metadata", {})
    if not isinstance(metadata, dict):
        raise HTTPException(status_code=400, detail="Metadata must be an object")
    
    # Проверяем обязательные поля
    if not metadata.get("name"):
        raise HTTPException(status_code=400, detail="Name is required in metadata")
    
    # Если есть validator, используем его
    if validator:
        try:
            validation_result = validator.validate_nft_metadata(metadata)
            if not validation_result["is_valid"]:
                raise HTTPException(status_code=400, detail=validation_result["error"])
            data["metadata"] = validation_result["sanitized"]
        except Exception as e:
            logger.warning(f"Ошибка валидации: {e}")
    
    return data

# Lifespan с упрощенной логикой
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Запуск музыкального сервера...")
    
    # Проверяем доступность базы данных
    if db_manager:
        try:
            await db_manager.create_pool()
            logger.info("✅ База данных подключена")
        except Exception as e:
            logger.warning(f"⚠️ База данных недоступна: {e}")
    
    logger.info("🎵 NFT Music Generator API готов к работе!")
    
    yield
    
    logger.info("🛑 Завершение работы сервера...")
    
    if db_manager:
        try:
            await db_manager.close_pool()
        except Exception as e:
            logger.error(f"Ошибка закрытия БД: {e}")

# Создание приложения
app = FastAPI(
    title="NFT Music Generator API",
    description="API для потоковой генерации музыки из NFT метаданных с аутентификацией через сессии",
    version="2.1.0",
    lifespan=lifespan,
    docs_url="/docs" if server_config.debug else None,
    redoc_url="/redoc" if server_config.debug else None,
)

# Добавляем CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Эндпоинты
@app.get("/")
async def root():
    return {
        "message": "NFT Streaming Music Generator API",
        "status": "running",
        "version": "2.1.0",
        "authentication": "session-based",
        "platform": "Timeweb Cloud",
        "streaming": True,
        "format": "WAV",
        "port": server_config.port
    }

@app.get("/health")
async def health_check():
    db_status = "not_available"
    if db_manager:
        try:
            samples = await db_manager.fetch_samples()
            db_status = "connected" if samples else "no_data"
        except Exception:
            db_status = "error"
    
    return {
        "status": "healthy",
        "database": db_status,
        "streaming_enabled": True,
        "session_auth": True,
        "timestamp": time.time(),
        "port": server_config.port
    }

@app.post("/generate-music-stream")
async def generate_music_stream_endpoint(
    request: Request,
    session_data: dict = Depends(verify_session_token)
):
    try:
        # Валидация запроса
        request_data = await request.json()
        validated_data = validate_music_request(request_data)
        
        user_address = session_data["address"]
        logger.info(f"🎵 Генерация музыки для пользователя: {user_address}")
        logger.info(f"📝 NFT: {validated_data['metadata'].get('name', 'Unknown')}")
        
        # Если доступны полные модули - используем их
        if streaming_music_generator and audio_downloader and db_manager:
            try:
                logger.info("🔍 Загружаем сэмплы из базы данных...")
                samples = await db_manager.fetch_samples()
                
                logger.info("📦 Скачиваем аудио файлы для NFT...")
                file_info = await audio_downloader.download_nft_audio_files(
                    validated_data["metadata"], samples
                )
                
                if file_info:
                    logger.info("🎼 Генерируем музыку из сэмплов...")
                    return StreamingResponse(
                        streaming_music_generator.generate_music_stream(
                            validated_data["metadata"], file_info
                        ),
                        media_type="audio/wav",
                        headers={
                            "Content-Disposition": "inline; filename=nft_music.wav",
                            "Cache-Control": "no-cache",
                            "X-Generated-For": hashlib.sha256(user_address.encode()).hexdigest()[:8],
                        }
                    )
                else:
                    logger.warning("⚠️ Не удалось загрузить аудио файлы")
                    
            except Exception as e:
                logger.error(f"❌ Ошибка полной генерации: {e}")                
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Ошибка генерации музыки: {e}")
        raise HTTPException(status_code=500, detail="Ошибка генерации музыки")

@app.get("/samples")
async def get_samples(session_data: dict = Depends(verify_session_token)):
    """Получение списка доступных сэмплов (требует аутентификации)"""
    if not db_manager:
        return {"count": 0, "samples": [], "status": "database_not_available"}
    
    try:
        samples = await db_manager.fetch_samples()
        user_address = session_data["address"]
        
        logger.info(f"📊 Пользователь {user_address} запросил список сэмплов")
        
        return {
            "count": len(samples), 
            "samples": samples,
            "user": hashlib.sha256(user_address.encode()).hexdigest()[:8]
        }
    except Exception as e:
        logger.error(f"❌ Ошибка получения сэмплов: {e}")
        return {"count": 0, "samples": [], "error": str(e)}

@app.get("/session/info")
async def get_session_info(session_data: dict = Depends(verify_session_token)):
    """Получение информации о текущей сессии"""
    return {
        "user": hashlib.sha256(session_data["address"].encode()).hexdigest()[:8],
        "domain": session_data["domain"],
        "timestamp": session_data["timestamp"],
        "expires_in": max(0, 3600 - (int(time.time()) - session_data["timestamp"])),
        "valid": True
    }

# Обработчик 404
@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return JSONResponse(
        status_code=404,
        content={"detail": "Endpoint not found", "available_endpoints": ["/", "/health", "/generate-music-stream", "/samples", "/session/info"]}
    )

# Обработчик ошибок аутентификации
@app.exception_handler(401)
async def auth_error_handler(request: Request, exc):
    return JSONResponse(
        status_code=401,
        content={
            "detail": "Authentication required", 
            "info": "Obtain session token from main backend /api/session/create"
        }
    )

# Главная функция запуска
def main():
    logger.info(f"🚀 Запуск музыкального сервера на {server_config.host}:{server_config.port}")
    logger.info(f"🔗 Node.js сервер: {NODE_SERVER_URL}")
    logger.info("🔐 Режим: Аутентификация через сессии")
    logger.info("🎵 Поддержка: Потоковая генерация музыки")
    
    # Убеждаемся, что порт корректный
    port = server_config.port
    if port <= 0 or port > 65535:
        port = 8000
        logger.warning(f"Некорректный порт, используем {port}")
    
    uvicorn.run(
        "main:app",
        host=server_config.host,
        port=port,
        reload=False,
        log_level="info",
        access_log=True
    )

if __name__ == "__main__":
    main()