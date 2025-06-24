import aiohttp
import asyncio
from io import BytesIO
from typing import List, Tuple, Dict, Any
from config import ipfs_config

class AudioDownloader:
    """Класс для скачивания аудио файлов с IPFS"""
    
    def __init__(self):
        self.base_url = ipfs_config.base_url
    
    async def fetch_audio_data(self, session: aiohttp.ClientSession, url: str) -> BytesIO:
        """Асинхронно загружает аудиофайл в память"""
        max_retries = 3
        retry_delay = 2  # секунд
        for attempt in range(max_retries):
            try:
                async with session.get(url, timeout=30) as response:
                    response.raise_for_status()
                    file_data = await response.read()
                    print(f"Загружен файл: {url} (размер: {len(file_data)} байт)")
                    return BytesIO(file_data)
            except Exception as e:
                print(f"Ошибка загрузки {url} (попытка {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                else:
                    raise

    async def download_nft_audio_files(self, nft_metadata: Dict[str, Any], samples: List[Dict[str, Any]]) -> List[Tuple[BytesIO, List[int], int, float]]:
        """
        Загружает аудиофайлы для NFT
        
        Args:
            nft_metadata: Метаданные NFT
            samples: Список всех доступных сэмплов
            
        Returns:
            List[Tuple[BytesIO, List[int], int, float]]: 
            Список кортежей (аудио_данные, паттерн, биты, bpm_сэмпла)
        """
        files_to_process = []
        attributes = nft_metadata.get("attributes", [])
        
        # Атрибуты, которые не являются сэмплами
        non_sample_traits = {"Feature", "Bpm", "Bassboost"}
        
        for attr in attributes:
            trait_type = attr.get("trait_type")
            value = attr.get("value")
            
            # Пропускаем вокал "Probably_Nothing"
            if trait_type == "Vocal" and value == "Probably_Nothing":
                print(f"Пропущен атрибут: trait_type={trait_type}, value={value} (вокал отсутствует)")
                continue
            
            # Пропускаем служебные атрибуты
            if trait_type in non_sample_traits:
                print(f"Пропущен атрибут: trait_type={trait_type}, value={value} (не является сэмплом)")
                continue
            
            # Ищем сэмпл в базе данных
            sample = next((s for s in samples if s["sample_type"] == trait_type and s["sample_name"] == value), None)
            if sample:
                cid = sample["sample_cid"]
                files_to_process.append((
                    f"{self.base_url}{cid}",
                    sample["sample_pattern"],
                    sample["sample_beats"],
                    sample["sample_bpm"]
                ))
                print(f"Добавлен сэмпл: trait_type={trait_type}, value={value}, cid={cid}")
            else:
                print(f"Предупреждение: сэмпл для trait_type={trait_type}, value={value} не найден в базе.")
        
        if not files_to_process:
            print(f"Ошибка: для NFT не найдено ни одного сэмпла.")
            return []
        
        # Скачиваем все файлы асинхронно
        async with aiohttp.ClientSession() as session:
            tasks = [self.fetch_audio_data(session, url) for url, _, _, _ in files_to_process]
            audio_data = await asyncio.gather(*tasks, return_exceptions=True)
        
        print(f"Длина audio_data: {len(audio_data)}, длина files_to_process: {len(files_to_process)}")
        
        # Обрабатываем результаты
        result = []
        for i, (data, (url, pattern, beats, sample_bpm)) in enumerate(zip(audio_data, files_to_process)):
            if isinstance(data, Exception):
                print(f"Ошибка загрузки файла {url}: {data}")
                continue
            result.append((data, pattern, beats, sample_bpm))
        
        if not result:
            print(f"Ошибка: не удалось загрузить ни один файл для NFT.")
        
        return result

# Глобальный экземпляр загрузчика
audio_downloader = AudioDownloader()