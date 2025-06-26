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
import os
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from starlette.middleware.sessions import SessionMiddleware
import secrets
import hashlib
import time
import logging
import sys

# Настройка логирования для Timeweb Cloud
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Импорты с обработкой ошибок
try:
    from database import db_manager
    from audio_downloader import audio_downloader
    from music_generator import streaming_music_generator
    from config import server_config
    from security_config import SecuritySettings, SecurityHeadersMiddleware
    from input_validator import InputValidator
    from secure_logger import SecureLogger
    from error_handler import SecureErrorHandler
    from rate_limiter import create_rate_limit_dependency, cleanup_task
    logger.info("✅ Все модули успешно импортированы")
except ImportError as e:
    logger.error(f"❌ Ошибка импорта: {e}")
    raise

# Инициализация
limiter = Limiter(key_func=get_remote_address)
security_settings = SecuritySettings()

# Настройки API
NODE_SERVER_URL = os.getenv("NODE_SERVER_URL", "http://localhost:3000")
API_KEY = os.getenv("API_KEY", "your-secret-key")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# CSRF Protection
class CSRFProtection:
    def __init__(self):
        self.tokens: Dict[str, float] = {}
        self.secret_key = secrets.token_urlsafe(32)
    
    def generate_csrf_token(self, session_id: str) -> str:
        timestamp = str(int(time.time()))
        token_data = f"{session_id}:{timestamp}:{self.secret_key}"
        token = hashlib.sha256(token_data.encode()).hexdigest()
        self.tokens[token] = time.time() + 3600
        return token
    
    def validate_csrf_token(self, token: str, session_id: str) -> bool:
        if token not in self.tokens:
            return False
        if time.time() > self.tokens[token]:
            del self.tokens[token]
            return False
        return True
    
    def cleanup_expired_tokens(self):
        current_time = time.time()
        expired_tokens = [token for token, expiry in self.tokens.items() if current_time > expiry]
        for token in expired_tokens:
            del self.tokens[token]

csrf_protection = CSRFProtection()

# Валидация и безопасность
async def verify_csrf_token(request: Request, x_csrf_token: Optional[str] = Header(None)):
    if request.method in ["GET", "HEAD", "OPTIONS"]:
        return True
    
    public_endpoints = ["/health", "/stream-info", "/csrf-token"]
    if any(request.url.path.startswith(endpoint) for endpoint in public_endpoints):
        return True
    
    if not x_csrf_token:
        raise HTTPException(status_code=403, detail="CSRF token missing")
    
    session_id = request.session.get("session_id", "")
    if not csrf_protection.validate_csrf_token(x_csrf_token, session_id):
        raise HTTPException(status_code=403, detail="Invalid CSRF token")
    
    return True

async def validate_nft_request(request: Request) -> dict:
    try:
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > security_settings.MAX_REQUEST_SIZE:
            raise HTTPException(status_code=413, detail="Request too large")
        
        request_data = await request.json()
        
        if not isinstance(request_data, dict):
            raise HTTPException(status_code=400, detail="Request must be a JSON object")
        
        metadata = request_data.get("metadata", {})
        validation_result = InputValidator.validate_nft_metadata(metadata)
        
        if not validation_result["is_valid"]:
            raise HTTPException(status_code=400, detail=f"Invalid NFT metadata: {validation_result['error']}")
        
        request_data["metadata"] = validation_result["sanitized"]
        
        index = request_data.get("index")
        if index is not None:
            if not isinstance(index, int) or index < 0 or index > 1000000:
                raise HTTPException(status_code=400, detail="Invalid NFT index")
        
        return request_data
        
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON format")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=500, detail="Request validation failed")

async def verify_api_key(api_key: str = Depends(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key

async def verify_music_api_key(x_music_api_key: Optional[str] = Header(None)):
    if not x_music_api_key:
        raise HTTPException(status_code=401, detail="Музыкальный API ключ не предоставлен")
    
    if not InputValidator.validate_api_key(x_music_api_key):
        raise HTTPException(status_code=401, detail="Недействительный формат API ключа")
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{NODE_SERVER_URL}/api/validateMusicApiKey",
                json={"apiKey": x_music_api_key},
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    if result.get("valid"):
                        return result
                    else:
                        raise HTTPException(status_code=401, detail="Недействительный API ключ")
                else:
                    raise HTTPException(status_code=response.status, detail="Ошибка проверки API ключа")
    except aiohttp.ClientError:
        raise HTTPException(status_code=503, detail="Сервис проверки ключей недоступен")
    except Exception as e:
        logger.error(f"API key validation error: {e}")
        raise HTTPException(status_code=500, detail="Ошибка проверки API ключа")

class NFTMusicRequest:
    def __init__(self, data: Dict[str, Any]):
        self.metadata = data.get("metadata", {})
        self.attributes = self.metadata.get("attributes", [])
        self.index = data.get("index")

# Генерация музыки
async def generate_and_stream_music(nft_request: NFTMusicRequest) -> Iterator[bytes]:
    try:
        logger.info(f"Начинаем потоковую генерацию музыки для NFT index: {nft_request.index}")
        
        samples = await db_manager.fetch_samples()
        if not samples:
            raise HTTPException(status_code=500, detail="Не удалось получить сэмплы из базы данных")
        
        file_info = await audio_downloader.download_nft_audio_files(nft_request.metadata, samples)
        if not file_info:
            raise HTTPException(status_code=400, detail="Не найдено аудиофайлов для данного NFT")
        
        async for chunk in streaming_music_generator.generate_music_stream(nft_request.metadata, file_info):
            yield chunk
            
    except Exception as e:
        logger.error(f"Ошибка при генерации потоковой музыки: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Запуск сервера на Timeweb Cloud...")
    
    try:
        logger.info("📊 Подключение к базе данных...")
        await db_manager.create_pool()
        logger.info("✅ Подключение к базе данных установлено")
        
        health_ok = await db_manager.health_check()
        if health_ok:
            logger.info("💚 База данных работает корректно")
        else:
            logger.warning("⚠️ Проблемы с базой данных, но продолжаем запуск")
        
    except Exception as e:
        logger.error(f"❌ Ошибка подключения к базе данных: {e}")
        logger.info("🔄 Продолжаем запуск без базы данных")
    
    try:
        cleanup_task_handle = asyncio.create_task(cleanup_task())
        logger.info("🧹 Задача очистки rate limiter запущена")
        
        async def csrf_cleanup():
            while True:
                try:
                    await asyncio.sleep(300)
                    csrf_protection.cleanup_expired_tokens()
                except Exception as e:
                    logger.error(f"Ошибка очистки CSRF токенов: {e}")
        
        csrf_task_handle = asyncio.create_task(csrf_cleanup())
        logger.info("🔐 Задача очистки CSRF токенов запущена")
        
        logger.info("🎵 NFT Music Generator API готов к работе!")
        
    except Exception as e:
        logger.error(f"Ошибка запуска фоновых задач: {e}")
    
    yield
    
    logger.info("🛑 Завершение работы сервера...")
    
    try:
        cleanup_task_handle.cancel()
        csrf_task_handle.cancel()
        await db_manager.close_pool()
        logger.info("✅ Соединения с базой данных закрыты")
    except Exception as e:
        logger.error(f"Ошибка при завершении работы: {e}")
    
    logger.info("👋 Сервер остановлен")

# Создание приложения
app = FastAPI(
    title="NFT Music Generator API",
    description="API для потоковой генерации музыки из NFT метаданных",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs" if server_config.debug else None,
    redoc_url="/redoc" if server_config.debug else None,
)

# Middleware
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(SessionMiddleware, 
    secret_key=os.getenv("SESSION_SECRET", secrets.token_urlsafe(32)),
    https_only=not server_config.debug,
    max_age=3600
)
app.add_middleware(TrustedHostMiddleware, 
    allowed_hosts=["*"]  # Настройте под ваш домен
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=security_settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=[
        "Accept", "Accept-Language", "Content-Language", "Content-Type",
        "Authorization", "X-API-Key", "X-Music-Api-Key", "X-CSRF-Token", "X-Requested-With"
    ],
)

# Обработчики ошибок
@app.exception_handler(RateLimitExceeded)
async def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"detail": "Слишком много запросов. Пожалуйста, попробуйте позже."},
        headers={"Retry-After": str(exc.detail.retry_after)}
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
        "auth_required": True
    }

@app.get("/health")
async def health_check():
    try:
        samples = await db_manager.fetch_samples()
        db_status = "connected" if samples else "no_data"
    except Exception as e:
        db_status = f"error: {str(e)}"
    
    node_server_status = "unknown"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{NODE_SERVER_URL}/", timeout=aiohttp.ClientTimeout(total=5)) as response:
                node_server_status = "connected" if response.status == 200 else f"error_{response.status}"
    except Exception as e:
        node_server_status = f"error: {str(e)}"
    
    return {
        "status": "healthy",
        "database": db_status,
        "node_server": node_server_status,
        "streaming_enabled": True,
        "timestamp": asyncio.get_event_loop().time()
    }

@app.get("/csrf-token")
async def get_csrf_token(request: Request):
    session_id = request.session.get("session_id")
    if not session_id:
        session_id = secrets.token_urlsafe(16)
        request.session["session_id"] = session_id
    
    csrf_token = csrf_protection.generate_csrf_token(session_id)
    return {"csrf_token": csrf_token}

@app.post("/generate-music-stream")
async def generate_music_stream(
    request: Request,
    rate_check: bool = Depends(create_rate_limit_dependency(5, 60)),
    auth_data: dict = Depends(verify_music_api_key),
    csrf_valid: bool = Depends(verify_csrf_token)
):
    try:
        request_data = await validate_nft_request(request)
        nft_request = NFTMusicRequest(request_data)
        
        user_address = auth_data.get("address", "unknown")
        logger.info(f"Генерация музыки для пользователя: {user_address[:10]}...")
        
        return StreamingResponse(
            generate_and_stream_music(nft_request),
            media_type="audio/wav",
            headers={
                "Content-Disposition": f"inline; filename=nft_{nft_request.index or 'music'}.wav",
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0",
                "Accept-Ranges": "bytes",
                "Transfer-Encoding": "chunked",
                "X-Streaming": "true",
                **SecuritySettings.SECURITY_HEADERS
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Music generation error: {e}")
        raise HTTPException(status_code=500, detail="Ошибка генерации музыки")

@app.get("/samples")
@limiter.limit("1/minute")
async def get_samples(request: Request, api_key: str = Depends(verify_api_key)):
    try:
        samples = await db_manager.fetch_samples()
        return {
            "count": len(samples),
            "samples": samples
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка получения сэмплов: {str(e)}")

@app.get("/stream-info")
async def get_stream_info():
    return {
        "streaming_enabled": True,
        "supported_formats": ["WAV"],
        "chunk_size": "variable",
        "real_time_processing": True,
        "auth_required": True,
        "auth_header": "X-Music-Api-Key",
        "endpoints": {
            "streaming": "/generate-music-stream",
            "full_file": "/generate-music",
            "save_file": "/generate-music-file"
        }
    }

if __name__ == "__main__":
    logger.info(f"🚀 Запуск потокового сервера на {server_config.host}:{server_config.port}")
    logger.info(f"📁 Директория для выходных файлов: {server_config.output_dir}")
    logger.info(f"🔗 Node.js сервер для проверки ключей: {NODE_SERVER_URL}")
    logger.info("🎵 Режим: Потоковая генерация музыки с проверкой ключей")
    
    uvicorn.run(
        "main:app",
        host=server_config.host,
        port=server_config.port,
        reload=False,
        log_level="info",
        access_log=True,
        use_colors=True
    )