from fastapi import FastAPI, HTTPException, Request, Depends, Header
from fastapi.responses import StreamingResponse
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

from database import db_manager
from audio_downloader import audio_downloader
from music_generator import streaming_music_generator
from config import server_config

# URL Node.js сервера для проверки API ключей
NODE_SERVER_URL = os.getenv("NODE_SERVER_URL", "http://localhost:3000")

# Настройка API-ключа (оставляем для других эндпоинтов)
API_KEY = os.getenv("API_KEY", "your-secret-key")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(api_key: str = Depends(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key

async def verify_music_api_key(x_music_api_key: Optional[str] = Header(None)):
    """Проверяет валидность музыкального API ключа через Node.js сервер"""
    if not x_music_api_key:
        raise HTTPException(status_code=401, detail="Музыкальный API ключ не предоставлен")
    
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
                        raise HTTPException(status_code=401, detail=result.get("error", "Недействительный API ключ"))
                else:
                    error_data = await response.json()
                    raise HTTPException(
                        status_code=response.status, 
                        detail=error_data.get("error", "Ошибка проверки API ключа")
                    )
    except aiohttp.ClientError as e:
        print(f"Ошибка соединения с Node.js сервером: {e}")
        raise HTTPException(status_code=503, detail="Сервис проверки ключей недоступен")
    except Exception as e:
        print(f"Неожиданная ошибка при проверке API ключа: {e}")
        raise HTTPException(status_code=500, detail="Ошибка проверки API ключа")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Управление жизненным циклом приложения"""
    print("Запуск сервера...")
    await db_manager.create_pool()
    print("Подключение к базе данных установлено")
    
    yield
    
    print("Завершение работы сервера...")
    await db_manager.close_pool()
    print("Соединения с базой данных закрыты")

app = FastAPI(
    title="NFT Music Generator API",
    description="API для потоковой генерации музыки из NFT метаданных",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins= os.getenv("FRONTEND_URL", "http://localhost:5173"),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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

@app.post("/generate-music-stream")
async def generate_music_stream(
    request: Request,
    auth_data: dict = Depends(verify_music_api_key)
):
    """
    Эндпоинт для потоковой генерации и передачи музыки чанками
    Требует валидный музыкальный API ключ в заголовке X-Music-Api-Key
    """
    try:
        request_data = await request.json()
        nft_request = NFTMusicRequest(request_data)
        
        user_address = auth_data.get("address")
        print(f"Получен запрос на потоковую генерацию музыки для NFT index: {nft_request.index} от пользователя: {user_address}")
        
        return StreamingResponse(
            generate_and_stream_music(nft_request),
            media_type="audio/wav",
            headers={
                "Content-Disposition": f"inline; filename=nft_{nft_request.index or 'music'}.wav",
                "Cache-Control": "no-cache",
                "Accept-Ranges": "bytes",
                "Transfer-Encoding": "chunked",
                "X-Streaming": "true",
                "X-User-Address": user_address
            }
        )
        
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Некорректный JSON в запросе")
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Отсутствует обязательное поле: {str(e)}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"Ошибка при потоковой генерации музыки: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Внутренняя ошибка сервера: {str(e)}")

@app.post("/generate-music")
async def generate_music(
    request: Request,
    auth_data: dict = Depends(verify_music_api_key)
):
    """
    Альтернативный эндпоинт для полной генерации музыки
    Требует валидный музыкальный API ключ
    """
    try:
        request_data = await request.json()
        nft_request = NFTMusicRequest(request_data)
        
        user_address = auth_data.get("address")
        print(f"Получен запрос на полную генерацию музыки для NFT index: {nft_request.index} от пользователя: {user_address}")
        
        # Получаем сэмплы из базы данных
        samples = await db_manager.fetch_samples()
        if not samples:
            raise HTTPException(status_code=500, detail="Не удалось получить сэмплы из базы данных")
        
        # Загружаем аудиофайлы
        file_info = await audio_downloader.download_nft_audio_files(nft_request.metadata, samples)
        if not file_info:
            raise HTTPException(status_code=400, detail="Не найдено аудиофайлов для данного NFT")
        
        # Генерируем музыку (неотемовая версия)
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
                "Cache-Control": "no-cache",
                "Content-Length": str(len(audio_buffer.getvalue())),
                "X-User-Address": user_address
            }
        )
        
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Некорректный JSON в запросе")
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Отсутствует обязательное поле: {str(e)}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"Ошибка при генерации музыки: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Внутренняя ошибка сервера: {str(e)}")

@app.post("/generate-music-file")
async def generate_music_file(
    request: Request,
    auth_data: dict = Depends(verify_music_api_key)
):
    """
    Эндпоинт для генерации и сохранения музыки в файл
    Требует валидный музыкальный API ключ
    """
    try:
        request_data = await request.json()
        nft_request = NFTMusicRequest(request_data)
        
        user_address = auth_data.get("address")
        print(f"Получен запрос на создание файла для NFT index: {nft_request.index} от пользователя: {user_address}")
        
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
        
        # Сохраняем в файл
        filename = f"nft_{nft_request.index or 'unknown'}_{user_address[-6:]}.wav"
        filepath = await streaming_music_generator.save_to_file(audio_buffer, filename)
        
        return {
            "message": "Музыка сгенерирована и сохранена",
            "file_path": filepath,
            "file_size": len(audio_buffer.getvalue()),
            "nft_index": nft_request.index,
            "user_address": user_address,
            "format": "WAV"
        }
        
    except Exception as e:
        print(f"Ошибка при создании файла: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Внутренняя ошибка сервера: {str(e)}")

@app.get("/samples")
async def get_samples(api_key: str = Depends(verify_api_key)):
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