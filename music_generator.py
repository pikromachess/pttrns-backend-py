import asyncio
import uuid
import numpy as np
import soundfile as sf
from pathlib import Path
from io import BytesIO
from typing import List, Tuple, Dict, Any, Iterator
from audio_processor import streaming_audio_processor
from config import audio_config, server_config

class StreamingMusicGenerator:
    """Класс для потоковой генерации музыки из NFT метаданных"""
    
    def __init__(self):
        self.audio_processor = streaming_audio_processor
        self.config = audio_config
        self.server_config = server_config
        self.target_samplerate = 44100  # Стандартная частота дискретизации
    
    async def generate_music_stream(self, nft_metadata: Dict[str, Any], file_info: List[Tuple[BytesIO, List[int], int, float]]) -> Iterator[bytes]:
        """
        Генерирует музыку потоково
        
        Args:
            nft_metadata: Метаданные NFT
            file_info: Список кортежей (аудио_данные, паттерн, биты, bpm_сэмпла)
            
        Yields:
            bytes: Чанки аудио данных в формате WAV
        """
        try:
            if not file_info:
                raise ValueError("Нет аудиофайлов для объединения")
            
            # Извлекаем параметры из метаданных NFT
            attributes = nft_metadata.get("attributes", [])
            bpm = self._extract_bpm(attributes)
            bassboost = self._extract_bassboost(attributes)
            
            print(f"Генерируем NFT потоково: bpm={bpm}, bassboost={bassboost}")
            
            # Генерируем объединенное аудио
            combined_audio_data, samplerate = await self._combine_audio_streaming(file_info, bpm)
            
            # Применяем эффекты
            if bassboost:
                print("Применяем bassboost")
                combined_audio_data = self.audio_processor.apply_bassboost_streaming(
                    combined_audio_data, samplerate
                )
            
            print("Применяем soft clipper")
            combined_audio_data = self.audio_processor.apply_soft_clipper_streaming(combined_audio_data)            
                       
            # Генерируем чанки для потоковой передачи
            print(f"Начинаем потоковую передачу, размер аудио: {combined_audio_data.shape}")
            
            for chunk in self.audio_processor.create_audio_chunks(combined_audio_data, samplerate):
                yield chunk
                
        except Exception as e:
            print(f"Ошибка при потоковой генерации музыки: {str(e)}")
            # Возвращаем ошибку как аудио данные (тишина)
            error_audio = np.zeros((self.target_samplerate, 2), dtype=np.float32)  # 1 секунда тишины
            for chunk in self.audio_processor.create_audio_chunks(error_audio, self.target_samplerate):
                yield chunk
    
    async def generate_music_from_nft(self, nft_metadata: Dict[str, Any], file_info: List[Tuple[BytesIO, List[int], int, float]]) -> BytesIO:
        """
        Генерирует музыку из NFT метаданных (неотемовая версия для совместимости)
        
        Args:
            nft_metadata: Метаданные NFT
            file_info: Список кортежей (аудио_данные, паттерн, биты, bpm_сэмпла)
            
        Returns:
            BytesIO: Сгенерированный аудио файл в памяти
        """
        if not file_info:
            raise ValueError("Нет аудиофайлов для объединения")
        
        # Извлекаем параметры из метаданных NFT
        attributes = nft_metadata.get("attributes", [])
        bpm = self._extract_bpm(attributes)
        bassboost = self._extract_bassboost(attributes)
        
        print(f"Генерируем NFT: bpm={bpm}, bassboost={bassboost}")
        
        # Объединяем аудиофайлы
        combined_audio_data, samplerate = await self._combine_audio_streaming(file_info, bpm)
        
        # Применяем эффекты
        if bassboost:
            print("Применяем bassboost")
            combined_audio_data = self.audio_processor.apply_bassboost_streaming(
                combined_audio_data, samplerate
            )
        
        print("Применяем soft clipper")
        combined_audio_data = self.audio_processor.apply_soft_clipper_streaming(combined_audio_data)
                       
        # Экспортируем в память
        output_buffer = BytesIO()
        sf.write(output_buffer, combined_audio_data, samplerate, format='WAV')
        output_buffer.seek(0)
        
        print(f"Музыка сгенерирована успешно, размер: {len(output_buffer.getvalue())} байт")
        return output_buffer
    
    def _extract_bpm(self, attributes: List[Dict[str, Any]]) -> float:
        """Извлекает BPM из атрибутов NFT"""
        bpm_attr = next((attr for attr in attributes if attr.get("trait_type") == "Bpm"), None)
        if bpm_attr:
            try:
                return float(bpm_attr["value"])
            except (ValueError, TypeError):
                print(f"Некорректное значение BPM: {bpm_attr['value']}, используем значение по умолчанию")
        return self.config.default_bpm
    
    def _extract_bassboost(self, attributes: List[Dict[str, Any]]) -> bool:
        """Извлекает настройку bassboost из атрибутов NFT"""
        bassboost_attr = next((attr for attr in attributes if attr.get("trait_type") == "Bassboost"), None)
        if bassboost_attr:
            return str(bassboost_attr["value"]).lower() == "true"
        return False
    
    async def _combine_audio_streaming(self, file_info: List[Tuple[BytesIO, List[int], int, float]], bpm: float) -> Tuple[np.ndarray, int]:
        """
        Асинхронно объединяет аудиофайлы с учетом их паттернов и BPM
        Возвращает numpy array и частоту дискретизации
        """
        beat_duration_samples = int(self.target_samplerate * 60 / bpm)
        
        # Загружаем и обрабатываем все треки
        processed_tracks = []
        max_duration_samples = 0
        
        for data, pattern, beats, sample_bpm in file_info:
            try:
                # Загружаем аудио
                audio_data, original_samplerate = self.audio_processor.load_audio_stream(data)
                
                # Ресэмплируем если нужно
                if original_samplerate != self.target_samplerate:
                    audio_data = self.audio_processor.resample_audio(
                        audio_data, original_samplerate, self.target_samplerate
                    )
                
                # Корректируем скорость под BPM
                speed_factor = bpm / sample_bpm
                if speed_factor != 1.0:
                    audio_data = self.audio_processor.change_speed(audio_data, speed_factor)
                
                # Приводим к стерео если нужно
                if len(audio_data.shape) == 1:  # моно -> стерео
                    audio_data = np.column_stack((audio_data, audio_data))
                
                # Вычисляем длительность одного цикла в сэмплах
                cycle_duration_samples = beat_duration_samples * beats
                
                # Обрезаем или зацикливаем аудио под длительность цикла
                if len(audio_data) > cycle_duration_samples:
                    audio_data = audio_data[:cycle_duration_samples]
                elif len(audio_data) < cycle_duration_samples:
                    # Зацикливаем аудио
                    loops_needed = int(np.ceil(cycle_duration_samples / len(audio_data)))
                    audio_data = np.tile(audio_data, (loops_needed, 1))[:cycle_duration_samples]
                
                # Рассчитываем общую длительность трека на основе паттерна
                pattern_length = len(pattern)
                track_duration_samples = pattern_length * cycle_duration_samples
                max_duration_samples = max(max_duration_samples, track_duration_samples)
                
                processed_tracks.append((audio_data, pattern, cycle_duration_samples))
                
            except Exception as e:
                print(f"Ошибка обработки трека: {e}")
                continue
        
        if not processed_tracks:
            raise ValueError("Не удалось обработать ни одного трека")
        
        # Создаем финальный микс
        print(f"Создаем микс длительностью {max_duration_samples} сэмплов")
        final_mix = np.zeros((max_duration_samples, 2), dtype=np.float32)
        
        for audio_data, pattern, cycle_duration_samples in processed_tracks:
            # Создаем трек полной длины
            track = np.zeros((max_duration_samples, 2), dtype=np.float32)
            
            # Применяем паттерн
            for i, active in enumerate(pattern):
                if active > 0:
                    start_sample = i * cycle_duration_samples
                    end_sample = start_sample + cycle_duration_samples
                    
                    if end_sample <= max_duration_samples:
                        # Накладываем аудио с учетом интенсивности
                        segment = audio_data * active
                        track[start_sample:end_sample] += segment
            
            # Добавляем трек к общему миксу
            final_mix += track
        
        # Нормализуем амплитуду чтобы избежать клиппинга
        max_amplitude = np.max(np.abs(final_mix))
        if max_amplitude > 0.95:
            final_mix = final_mix * (0.95 / max_amplitude)
        
        return final_mix, self.target_samplerate
    
    async def save_to_file(self, audio_buffer: BytesIO, filename: str = None) -> str:
        """
        Сохраняет аудио файл на диск
        
        Args:
            audio_buffer: Аудио данные в памяти
            filename: Имя файла (если не указано, генерируется автоматически)
            
        Returns:
            str: Путь к сохраненному файлу
        """
        if filename is None:
            filename = f"nft_{uuid.uuid4().hex[:8]}.wav"
        
        output_path = Path(self.server_config.output_dir) / filename
        output_path.parent.mkdir(exist_ok=True)
        
        # Читаем данные из буфера и записываем в файл
        audio_buffer.seek(0)
        with open(output_path, 'wb') as f:
            f.write(audio_buffer.read())
        
        print(f"Файл сохранен: {output_path}")
        return str(output_path)

# Глобальный генератор музыки
streaming_music_generator = StreamingMusicGenerator()