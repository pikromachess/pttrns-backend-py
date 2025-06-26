import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass
class DatabaseConfig:
    """Конфигурация базы данных"""    
    database_url: str = os.getenv("DATABASE_URL")   

@dataclass
class AudioConfig:
    """Конфигурация для обработки аудио"""
    target_lufs: float = -8.0
    bassboost_db: float = 6.0
    soft_clipper_threshold_db: float = -6.0
    default_bpm: float = 130.0
    output_format: str = "wav"    

@dataclass
class IPFSConfig:
    """Конфигурация для IPFS"""
    base_url: str = os.getenv("IPFS_BASE_URL", "https://salmon-key-halibut-684.mypinata.cloud/ipfs/")

@dataclass
class ServerConfig:
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8000"))
    output_dir: str = os.getenv("OUTPUT_DIR", "./output")
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"
    
    def __post_init__(self):
        # Создаем output директорию если не существует
        os.makedirs(self.output_dir, exist_ok=True)

# Глобальные конфигурации
db_config = DatabaseConfig()
audio_config = AudioConfig()
ipfs_config = IPFSConfig()
server_config = ServerConfig()