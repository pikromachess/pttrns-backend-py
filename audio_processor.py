import numpy as np
import soundfile as sf
from io import BytesIO
from scipy.signal import lfilter
from typing import Iterator, Tuple
from config import audio_config

class StreamingAudioProcessor:
    """Класс для потоковой обработки аудио"""
    
    def __init__(self):
        self.config = audio_config
        self.chunk_size = 4096  # Размер чанка для потоковой обработки
    
    def load_audio_stream(self, audio_buffer: BytesIO) -> Tuple[np.ndarray, int]:
        """Загружает аудио из буфера"""
        try:
            audio_buffer.seek(0)
            data, samplerate = sf.read(audio_buffer)
            print(f"Загружено аудио: samplerate={samplerate}, shape={data.shape}")
            return data, samplerate
        except Exception as e:
            print(f"Ошибка загрузки аудио: {e}")
            raise
    
    def create_audio_chunks(self, audio_data: np.ndarray, samplerate: int, chunk_duration_ms: int = 100) -> Iterator[bytes]:
        """Генерирует чанки аудио данных для потоковой передачи"""
        try:
            # Вычисляем размер чанка в сэмплах
            chunk_samples = int(samplerate * chunk_duration_ms / 1000)
            
            # Если аудио стерео, учитываем это
            if len(audio_data.shape) > 1:
                total_samples = audio_data.shape[0]
            else:
                total_samples = len(audio_data)
            
            # Создаем WAV заголовок
            buffer = BytesIO()
            sf.write(buffer, audio_data, samplerate, format='WAV')
            wav_data = buffer.getvalue()
            
            # Размер заголовка WAV (обычно 44 байта)
            header_size = 44
            header = wav_data[:header_size]
            audio_bytes = wav_data[header_size:]
            
            # Отправляем заголовок
            yield header
            
            # Отправляем аудио данные чанками
            chunk_byte_size = chunk_samples * audio_data.dtype.itemsize
            if len(audio_data.shape) > 1:  # стерео
                chunk_byte_size *= audio_data.shape[1]
            
            for i in range(0, len(audio_bytes), chunk_byte_size):
                chunk = audio_bytes[i:i + chunk_byte_size]
                if chunk:
                    yield chunk
                    
        except Exception as e:
            print(f"Ошибка создания чанков: {e}")
            raise
    
    def apply_bassboost_streaming(self, audio_data: np.ndarray, samplerate: int, boost_db: float = None) -> np.ndarray:
        """Применяет bassboost к аудио данным"""
        if boost_db is None:
            boost_db = self.config.bassboost_db
            
        try:
            # Если стерео, обрабатываем каждый канал
            if len(audio_data.shape) > 1 and audio_data.shape[1] == 2:
                left_channel = audio_data[:, 0]
                right_channel = audio_data[:, 1]
                
                filtered_left = self._apply_bass_filter(left_channel, samplerate, boost_db)
                filtered_right = self._apply_bass_filter(right_channel, samplerate, boost_db)
                
                return np.column_stack((filtered_left, filtered_right))
            else:
                return self._apply_bass_filter(audio_data, samplerate, boost_db)
                
        except Exception as e:
            print(f"Ошибка при применении bassboost: {e}")
            return audio_data
    
    def _apply_bass_filter(self, samples: np.ndarray, sample_rate: int, boost_db: float) -> np.ndarray:
        """Применяет low-shelf фильтр для усиления басов"""
        try:
            cutoff = 200  # Частота среза в Hz
            A = 10 ** (boost_db / 40)
            w0 = 2 * np.pi * cutoff / sample_rate
            cos_w0 = np.cos(w0)
            sin_w0 = np.sin(w0)
            alpha = sin_w0 / (2 * 1.0)
            
            # Коэффициенты low-shelf фильтра
            b0 = A * ((A + 1) - (A - 1) * cos_w0 + 2 * np.sqrt(A) * alpha)
            b1 = 2 * A * ((A - 1) - (A + 1) * cos_w0)
            b2 = A * ((A + 1) - (A - 1) * cos_w0 - 2 * np.sqrt(A) * alpha)
            a0 = (A + 1) + (A - 1) * cos_w0 + 2 * np.sqrt(A) * alpha
            a1 = -2 * ((A - 1) + (A + 1) * cos_w0)
            a2 = (A + 1) + (A - 1) * cos_w0 - 2 * np.sqrt(A) * alpha
            
            # Нормализуем коэффициенты
            b = np.array([b0, b1, b2]) / a0
            a = np.array([1.0, a1 / a0, a2 / a0])
            
            # Применяем фильтр
            return lfilter(b, a, samples)
            
        except Exception as e:
            print(f"Ошибка в bass фильтре: {e}")
            return samples
    
    def apply_soft_clipper_streaming(self, audio_data: np.ndarray, threshold_db: float = None) -> np.ndarray:
        """Применяет soft clipper к аудио данным"""
        if threshold_db is None:
            threshold_db = self.config.soft_clipper_threshold_db
            
        try:
            # Вычисляем порог (нормализованный к [-1, 1])
            threshold = 10 ** (threshold_db / 20.0)
            
            # Применяем soft clipping с tanh
            return np.tanh(audio_data / threshold) * threshold
            
        except Exception as e:
            print(f"Ошибка при применении soft clipper: {e}")
            return audio_data
        
    def resample_audio(self, audio_data: np.ndarray, original_rate: int, target_rate: int) -> np.ndarray:
        """Изменяет частоту дискретизации аудио"""
        try:
            if original_rate == target_rate:
                return audio_data
            
            # Простая ресэмплинг через интерполяцию
            ratio = target_rate / original_rate
            new_length = int(len(audio_data) * ratio)
            
            if len(audio_data.shape) > 1:  # стерео
                resampled = np.zeros((new_length, audio_data.shape[1]))
                for channel in range(audio_data.shape[1]):
                    resampled[:, channel] = np.interp(
                        np.linspace(0, len(audio_data) - 1, new_length),
                        np.arange(len(audio_data)),
                        audio_data[:, channel]
                    )
                return resampled
            else:  # моно
                return np.interp(
                    np.linspace(0, len(audio_data) - 1, new_length),
                    np.arange(len(audio_data)),
                    audio_data
                )
                
        except Exception as e:
            print(f"Ошибка ресэмплинга: {e}")
            return audio_data
    
    def change_speed(self, audio_data: np.ndarray, speed_factor: float) -> np.ndarray:
        """Изменяет скорость воспроизведения аудио"""
        try:
            if speed_factor == 1.0:
                return audio_data
                
            new_length = int(len(audio_data) / speed_factor)
            
            if len(audio_data.shape) > 1:  # стерео
                new_audio = np.zeros((new_length, audio_data.shape[1]))
                for channel in range(audio_data.shape[1]):
                    new_audio[:, channel] = np.interp(
                        np.linspace(0, len(audio_data) - 1, new_length),
                        np.arange(len(audio_data)),
                        audio_data[:, channel]
                    )
                return new_audio
            else:  # моно
                return np.interp(
                    np.linspace(0, len(audio_data) - 1, new_length),
                    np.arange(len(audio_data)),
                    audio_data
                )
                
        except Exception as e:
            print(f"Ошибка изменения скорости: {e}")
            return audio_data

# Глобальный экземпляр процессора
streaming_audio_processor = StreamingAudioProcessor()