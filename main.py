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

from database import db_manager
from audio_downloader import audio_downloader
from music_generator import streaming_music_generator
from config import server_config

from security_config import SecuritySettings, SecurityHeadersMiddleware
from input_validator import InputValidator
from secure_logger import SecureLogger
from error_handler import SecureErrorHandler
import logging

from rate_limiter import create_rate_limit_dependency, cleanup_task

limiter = Limiter(key_func=get_remote_address)
security_settings = SecuritySettings()

# URL Node.js сервера для проверки API ключей
NODE_SERVER_URL = os.getenv("NODE_SERVER_URL", "http://localhost:3000")

# Настройка API-ключа (оставляем для других эндпоинтов)
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
        self.tokens[token] = time.time() + 3600  # Действует 1 час
        return token
    
    def validate_csrf_token(self, token: str, session_id: str) -> bool:
        if token not in self.tokens:
            return False
        
        if time.time() > self.tokens[token]:
            del self.tokens[token]
            return False
        
        # Проверяем валидность токена
        for stored_token, expiry in list(self.tokens.items()):
            if time.time() > expiry:
                del self.tokens[stored_token]
        
        return token in self.tokens
    
    def cleanup_expired_tokens(self):
        current_time = time.time()
        expired_tokens = [token for token, expiry in self.tokens.items() if current_time > expiry]
        for token in expired_tokens:
            del self.tokens[token]

csrf_protection = CSRFProtection()

async def verify_csrf_token(request: Request, x_csrf_token: Optional[str] = Header(None)):
    # Пропускаем CSRF для GET запросов и некоторых публичных эндпоинтов
    if request.method in ["GET", "HEAD", "OPTIONS"]:
        return True
    
    # Пропускаем CSRF для определенных эндпоинтов (например, webhook'и)
    public_endpoints = ["/health", "/stream-info", "/csrf-token"]
    if any(request.url.path.startswith(endpoint) for endpoint in public_endpoints):
        return True
    
    if not x_csrf_token:
        await SecureLogger.security("csrf_token_missing", {"path": request.url.path})
        raise HTTPException(status_code=403, detail="CSRF token missing")
    
    session_id = request.session.get("session_id", "")
    if not csrf_protection.validate_csrf_token(x_csrf_token, session_id):
        await SecureLogger.security("csrf_token_invalid", {
            "path": request.url.path,
            "session_exists": bool(session_id)
        })
        raise HTTPException(status_code=403, detail="Invalid CSRF token")
    
    return True

# Добавьте обработчик для валидации NFT данных:
async def validate_nft_request(request: Request) -> dict:
    """Валидация запроса на генерацию музыки"""
    try:
        # Проверяем размер запроса
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > security_settings.MAX_REQUEST_SIZE:
            raise HTTPException(status_code=413, detail="Request too large")
        
        request_data = await request.json()
        
        # Валидация структуры запроса
        if not isinstance(request_data, dict):
            raise HTTPException(status_code=400, detail="Request must be a JSON object")
        
        # Валидация метаданных NFT
        metadata = request_data.get("metadata", {})
        validation_result = InputValidator.validate_nft_metadata(metadata)
        
        if not validation_result["is_valid"]:
            await SecureLogger.security("invalid_nft_metadata", {
                "error": validation_result["error"],
                "metadata_keys": list(metadata.keys()) if isinstance(metadata, dict) else "not_dict"
            })
            raise HTTPException(status_code=400, detail=f"Invalid NFT metadata: {validation_result['error']}")
        
        # Используем очищенные метаданные
        request_data["metadata"] = validation_result["sanitized"]
        
        # Валидация индекса
        index = request_data.get("index")
        if index is not None:
            if not isinstance(index, int) or index < 0 or index > 1000000:
                raise HTTPException(status_code=400, detail="Invalid NFT index")
        
        return request_data
        
    except json.JSONDecodeError:
        await SecureLogger.security("invalid_json_request", {"path": request.url.path})
        raise HTTPException(status_code=400, detail="Invalid JSON format")
    except HTTPException:
        raise  # Пробрасываем HTTP исключения как есть
    except Exception as e:
        error_info = SecureErrorHandler.handle_error(e, "nft_request_validation")
        await SecureLogger.error("nft_validation_unexpected_error", e, {"error_code": error_info["code"]})
        raise HTTPException(status_code=500, detail="Request validation failed")

async def verify_api_key(api_key: str = Depends(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key

async def verify_music_api_key(x_music_api_key: Optional[str] = Header(None)):
    """Проверяет валидность музыкального API ключа через Node.js сервер"""
    if not x_music_api_key:
        await SecureLogger.security("missing_api_key", {"endpoint": "music_api"})
        raise HTTPException(status_code=401, detail="Музыкальный API ключ не предоставлен")
    
    # Валидация формата API ключа
    if not InputValidator.validate_api_key(x_music_api_key):
        await SecureLogger.security("invalid_api_key_format", {"key_length": len(x_music_api_key)})
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
                        await SecureLogger.info("api_key_validated", {
                            "user_address": result.get("address", "unknown")[:10] + "..."
                        })
                        return result
                    else:
                        await SecureLogger.security("api_key_validation_failed", {
                            "error": result.get("error", "unknown")
                        })
                        raise HTTPException(status_code=401, detail="Недействительный API ключ")
                else:
                    await SecureLogger.error("api_validation_server_error", None, {
                        "status": response.status
                    })
                    raise HTTPException(status_code=response.status, detail="Ошибка проверки API ключа")
    except aiohttp.ClientError as e:
        await SecureLogger.error("nodejs_server_connection_failed", e)
        raise HTTPException(status_code=503, detail="Сервис проверки ключей недоступен")
    except Exception as e:
        error_info = SecureErrorHandler.handle_error(e, "api_key_validation")
        await SecureLogger.error("api_key_validation_error", e, {"error_code": error_info["code"]})
        raise HTTPException(status_code=500, detail="Ошибка проверки API ключа")
    
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Управление жизненным циклом приложения"""
    print("Запуск сервера...")
    await db_manager.create_pool()
    print("Подключение к базе данных установлено")
    
    # Запускаем задачу очистки rate limiter
    asyncio.create_task(cleanup_task())
    
    # Запускаем задачу очистки CSRF токенов
    async def csrf_cleanup():
        while True:
            await asyncio.sleep(300)  # Каждые 5 минут
            csrf_protection.cleanup_expired_tokens()
    
    asyncio.create_task(csrf_cleanup())
    
    await SecureLogger.info("server_started", {
        "version": "2.0.0",
        "security_enabled": True,
        "csrf_enabled": True,
        "rate_limiting_enabled": True
    })
    
    yield
    
    print("Завершение работы сервера...")
    await db_manager.close_pool()
    print("Соединения с базой данных закрыты")

app = FastAPI(
    title="NFT Music Generator API",
    description="API для потоковой генерации музыки из NFT метаданных",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs" if os.getenv("ENVIRONMENT") == "development" else None,  # Отключаем docs в продакшене
    redoc_url="/redoc" if os.getenv("ENVIRONMENT") == "development" else None,
)

app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(SessionMiddleware, 
    secret_key=os.getenv("SESSION_SECRET", secrets.token_urlsafe(32)),
    https_only=security_settings.SESSION_COOKIE_SECURE,
    max_age=3600  # 1 час
)
app.add_middleware(TrustedHostMiddleware, 
    allowed_hosts=["localhost", "your-domain.com", "*.your-domain.com"]
)

# Обновите CORS middleware:
app.add_middleware(
    CORSMiddleware,
    allow_origins=security_settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],  # Убираем "*"
    allow_headers=[
        "Accept",
        "Accept-Language", 
        "Content-Language",
        "Content-Type",
        "Authorization",
        "X-API-Key",
        "X-Music-Api-Key",
        "X-CSRF-Token",
        "X-Requested-With"
    ],  # Убираем "*"
)

# Обработчик ошибок лимита
@app.exception_handler(RateLimitExceeded)
async def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"detail": "Слишком много запросов. Пожалуйста, попробуйте позже."},
        headers={"Retry-After": str(exc.detail.retry_after)}
    )

class NFTMusicRequest:
    """Модель запроса для генерации музыки"""
    def __init__(self, data: Dict[str, Any]):
        self.metadata = data.get("metadata", {})
        self.attributes = self.metadata.get("attributes", [])
        self.index = data.get("index")

async def generate_and_stream_music(nft_request: NFTMusicRequest) -> Iterator[bytes]:
    """
    Асинхронный генератор для создания и потоковой передачи музыки
    """
    try:
        print(f"Начинаем потоковую генерацию музыки для NFT index: {nft_request.index}")
        
        # Получаем сэмплы из базы данных
        samples = await db_manager.fetch_samples()
        if not samples:
            raise HTTPException(status_code=500, detail="Не удалось получить сэмплы из базы данных")
        
        # Загружаем аудиофайлы
        file_info = await audio_downloader.download_nft_audio_files(nft_request.metadata, samples)
        if not file_info:
            raise HTTPException(status_code=400, detail="Не найдено аудиофайлов для данного NFT")
        
        # Генерируем музыку потоково
        async for chunk in streaming_music_generator.generate_music_stream(nft_request.metadata, file_info):
            yield chunk
            
    except Exception as e:
        print(f"Ошибка при генерации потоковой музыки: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    """Корневой эндпоинт для проверки работоспособности"""
    return {
        "message": "NFT Streaming Music Generator API",
        "status": "running",
        "version": "2.0.0",
        "streaming": True,
        "format": "WAV",
        "auth_required": True
    }

@app.get("/health")
async def health_check():
    """Проверка состояния сервиса"""
    try:
        samples = await db_manager.fetch_samples()
        db_status = "connected" if samples else "no_data"
    except Exception as e:
        db_status = f"error: {str(e)}"
    
    # Проверяем доступность Node.js сервера
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
    rate_check: bool = Depends(create_rate_limit_dependency(5, 60)),  # 5 запросов в минуту
    auth_data: dict = Depends(verify_music_api_key),
    csrf_valid: bool = Depends(verify_csrf_token)
):
    """
    Эндпоинт для потоковой генерации и передачи музыки чанками
    Требует валидный музыкальный API ключ в заголовке X-Music-Api-Key
    """
    try:
        # Безопасная валидация запроса
        request_data = await validate_nft_request(request)
        nft_request = NFTMusicRequest(request_data)
        
        user_address = auth_data.get("address", "unknown")
        
        await SecureLogger.user_action("music_generation_stream_start", user_address, {
            "nft_index": nft_request.index,
            "has_metadata": bool(nft_request.metadata)
        })
        
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
                # Удаляем X-User-Address для приватности
                **SecuritySettings.SECURITY_HEADERS
            }
        )
        
    except HTTPException:
        raise  # Пробрасываем HTTP исключения как есть
    except Exception as e:
        error_info = SecureErrorHandler.handle_error(e, "music_generation_stream")
        await SecureLogger.error("music_generation_stream_failed", e, {
            "error_code": error_info["code"],
            "user_address": auth_data.get("address", "unknown")[:10] + "..."
        })
        raise HTTPException(status_code=500, detail=error_info["message"])

@app.post("/generate-music")
async def generate_music(
    request: Request,
    rate_check: bool = Depends(create_rate_limit_dependency(3, 60)),  # Добавить rate limiting
    auth_data: dict = Depends(verify_music_api_key),
    csrf_valid: bool = Depends(verify_csrf_token)  # Добавить CSRF
):
    """
    Альтернативный эндпоинт для полной генерации музыки
    Требует валидный музыкальный API ключ
    """
    try:
        # ЗАМЕНИТЬ НА БЕЗОПАСНУЮ ВАЛИДАЦИЮ:
        request_data = await validate_nft_request(request)
        nft_request = NFTMusicRequest(request_data)
        
        user_address = auth_data.get("address", "unknown")
        
        await SecureLogger.user_action("music_generation_full_start", user_address, {
            "nft_index": nft_request.index,
            "has_metadata": bool(nft_request.metadata)
        })
        
        # Получаем сэмплы из базы данных
        samples = await db_manager.fetch_samples()
        if not samples:
            raise HTTPException(status_code=500, detail="Не удалось получить сэмплы из базы данных")
        
        # Загружаем аудиофайлы
        file_info = await audio_downloader.download_nft_audio_files(nft_request.metadata, samples)
        if not file_info:
            raise HTTPException(status_code=400, detail="Не найдено аудиофайлов для данного NFT")
        
        # Генерируем музыку
        audio_buffer = await streaming_music_generator.generate_music_from_nft(nft_request.metadata, file_info)
        
        def iterfile():
            """Генератор для стриминга файла"""
            audio_buffer.seek(0)
            while True:
                chunk = audio_buffer.read(8192)
                if not chunk:
                    break
                yield chunk
        
        return StreamingResponse(
            iterfile(),
            media_type="audio/wav",
            headers={
                "Content-Disposition": f"inline; filename=nft_{nft_request.index or 'music'}.wav",
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0",
                # Удаляем X-User-Address для приватности
                **SecuritySettings.SECURITY_HEADERS
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        error_info = SecureErrorHandler.handle_error(e, "music_generation_full")
        await SecureLogger.error("music_generation_full_failed", e, {
            "error_code": error_info["code"],
            "user_address": user_address[:10] + "..."
        })
        raise HTTPException(status_code=500, detail=error_info["message"])

@app.post("/generate-music-file")
async def generate_music_file(
    request: Request,
    rate_check: bool = Depends(create_rate_limit_dependency(2, 60)),  # Добавить rate limiting
    auth_data: dict = Depends(verify_music_api_key),
    csrf_valid: bool = Depends(verify_csrf_token)  # Добавить CSRF
):
    """
    Эндпоинт для генерации и сохранения музыки в файл
    Требует валидный музыкальный API ключ
    """
    try:
        # ЗАМЕНИТЬ НА БЕЗОПАСНУЮ ВАЛИДАЦИЮ:
        request_data = await validate_nft_request(request)
        nft_request = NFTMusicRequest(request_data)
        
        user_address = auth_data.get("address", "unknown")
        
        await SecureLogger.user_action("music_generation_file_start", user_address, {
            "nft_index": nft_request.index,
            "has_metadata": bool(nft_request.metadata)
        })
        
        # Получаем сэмплы из базы данных
        samples = await db_manager.fetch_samples()
        if not samples:
            raise HTTPException(status_code=500, detail="Не удалось получить сэмплы из базы данных")
        
        # Загружаем аудиофайлы
        file_info = await audio_downloader.download_nft_audio_files(nft_request.metadata, samples)
        if not file_info:
            raise HTTPException(status_code=400, detail="Не найдено аудиофайлов для данного NFT")
        
        # Генерируем музыку
        audio_buffer = await streaming_music_generator.generate_music_from_nft(nft_request.metadata, file_info)
        
        # Сохраняем в файл (безопасно)
        safe_filename = f"nft_{nft_request.index or 'unknown'}.wav"  # Убираем user_address
        filepath = await streaming_music_generator.save_to_file(audio_buffer, safe_filename)
        
        await SecureLogger.user_action("music_generation_file_success", user_address, {
            "file_size": len(audio_buffer.getvalue()),
            "nft_index": nft_request.index
        })
        
        return {
            "message": "Музыка сгенерирована и сохранена",
            "file_path": filepath,
            "file_size": len(audio_buffer.getvalue()),
            "nft_index": nft_request.index,
            "format": "WAV"
            # Убираем user_address из ответа для приватности
        }
        
    except HTTPException:
        raise
    except Exception as e:
        error_info = SecureErrorHandler.handle_error(e, "music_generation_file")
        await SecureLogger.error("music_generation_file_failed", e, {
            "error_code": error_info["code"],
            "user_address": user_address[:10] + "..."
        })
        raise HTTPException(status_code=500, detail=error_info["message"])

@app.get("/samples")
@limiter.limit("1/minute")
async def get_samples(request: Request, api_key: str = Depends(verify_api_key)):
    
    """Получить список всех доступных сэмплов (для отладки)"""
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
    """Получить информацию о возможностях потоковой передачи"""
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
    print(f"Запуск потокового сервера на {server_config.host}:{server_config.port}")
    print(f"Директория для выходных файлов: {server_config.output_dir}")
    print(f"Node.js сервер для проверки ключей: {NODE_SERVER_URL}")
    print("Режим: Потоковая генерация музыки с проверкой ключей")
    
    uvicorn.run(
        "main:app",
        host=server_config.host,
        port=server_config.port,
        reload=False,
        log_level="info",
        access_log=True,
        use_colors=True
    )