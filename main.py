import os
import sys
import logging
from fastapi import FastAPI, HTTPException, Request, Depends, Header
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
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

# Безопасные импорты
db_manager = safe_import()
audio_downloader, streaming_music_generator = safe_import_audio()
server_config = safe_import_config()

# Базовые настройки
NODE_SERVER_URL = os.getenv("NODE_SERVER_URL", "http://localhost:3000")
API_KEY = os.getenv("API_KEY", "your-secret-key")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# Упрощенная конфигурация CORS
ALLOWED_ORIGINS = [
    "https://pikromachess-pttrns-frontend-dc0f.twc1.net"
]

if os.getenv("ENVIRONMENT", "development") == "development":
    ALLOWED_ORIGINS.extend([
        "http://localhost:3000",
        "http://localhost:5173"
    ])

# Упрощенный rate limiter без внешних зависимостей
class SimpleRateLimiter:
    def __init__(self):
        self.requests = {}
    
    def check_limit(self, client_ip: str, max_requests: int = 5, window: int = 60) -> bool:
        current_time = time.time()
        
        if client_ip not in self.requests:
            self.requests[client_ip] = []
        
        # Удаляем старые запросы
        self.requests[client_ip] = [
            req_time for req_time in self.requests[client_ip] 
            if current_time - req_time < window
        ]
        
        if len(self.requests[client_ip]) >= max_requests:
            return False
        
        self.requests[client_ip].append(current_time)
        return True

rate_limiter = SimpleRateLimiter()

# Упрощенная валидация
def validate_nft_request_simple(data: dict) -> dict:
    """Упрощенная валидация без внешних модулей"""
    if not isinstance(data, dict):
        raise HTTPException(status_code=400, detail="Request must be a JSON object")
    
    metadata = data.get("metadata", {})
    if not isinstance(metadata, dict):
        raise HTTPException(status_code=400, detail="Metadata must be an object")
    
    # Проверяем обязательные поля
    if not metadata.get("name"):
        raise HTTPException(status_code=400, detail="Name is required")
    
    return data

# Упрощенная проверка API ключей
async def verify_music_api_key_simple(x_music_api_key: Optional[str] = Header(None)):
    if not x_music_api_key:
        raise HTTPException(status_code=401, detail="API ключ не предоставлен")
    
    # Упрощенная проверка формата
    if len(x_music_api_key) < 10:
        raise HTTPException(status_code=401, detail="Недействительный API ключ")
    
    return {"address": "validated_user", "valid": True}

# Lifespan с упрощенной логикой
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Запуск сервера на Timeweb Cloud...")
    
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
    description="API для потоковой генерации музыки из NFT метаданных",
    version="2.0.0",
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
        "version": "2.0.0",
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
        "timestamp": time.time(),
        "port": server_config.port
    }

@app.post("/generate-music-stream")
async def generate_music_stream_endpoint(
    request: Request,
    auth_data: dict = Depends(verify_music_api_key_simple)
):
    try:
        # Rate limiting
        client_ip = request.client.host if request.client else "unknown"
        if not rate_limiter.check_limit(client_ip):
            raise HTTPException(status_code=429, detail="Слишком много запросов")
        
        # Валидация запроса
        request_data = await request.json()
        validated_data = validate_nft_request_simple(request_data)
        
        logger.info(f"Генерация музыки для пользователя: {auth_data.get('address', 'unknown')}")
        
        # Если доступны полные модули - используем их
        if streaming_music_generator and audio_downloader and db_manager:
            try:
                samples = await db_manager.fetch_samples()
                file_info = await audio_downloader.download_nft_audio_files(
                    validated_data["metadata"], samples
                )
                
                if file_info:
                    return StreamingResponse(
                        streaming_music_generator.generate_music_stream(
                            validated_data["metadata"], file_info
                        ),
                        media_type="audio/wav",
                        headers={
                            "Content-Disposition": "inline; filename=nft_music.wav",
                            "Cache-Control": "no-cache",
                        }
                    )
            except Exception as e:
                logger.error(f"Ошибка полной генерации: {e}")
        
        # Fallback - простая генерация
        logger.info("Используем упрощенную генерацию музыки")
        return StreamingResponse(
            generate_simple_audio_stream(),
            media_type="audio/wav",
            headers={
                "Content-Disposition": "inline; filename=test_audio.wav",
                "Cache-Control": "no-cache",
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка генерации музыки: {e}")
        raise HTTPException(status_code=500, detail="Ошибка генерации музыки")

@app.get("/samples")
async def get_samples():
    if not db_manager:
        return {"count": 0, "samples": [], "status": "database_not_available"}
    
    try:
        samples = await db_manager.fetch_samples()
        return {"count": len(samples), "samples": samples}
    except Exception as e:
        logger.error(f"Ошибка получения сэмплов: {e}")
        return {"count": 0, "samples": [], "error": str(e)}

# Обработчик 404
@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return JSONResponse(
        status_code=404,
        content={"detail": "Endpoint not found"}
    )

# Главная функция запуска
def main():
    logger.info(f"🚀 Запуск сервера на {server_config.host}:{server_config.port}")
    logger.info(f"🔗 Node.js сервер: {NODE_SERVER_URL}")
    logger.info("🎵 Режим: Упрощенная потоковая генерация")
    
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