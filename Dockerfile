FROM python:3.11-slim

# Установка системных зависимостей для аудио обработки
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    libsndfile1-dev \
    && rm -rf /var/lib/apt/lists/*

# Создание не-root пользователя
RUN useradd -m -u 1000 appuser
USER appuser

# Создание рабочей директории
WORKDIR /app

# Копирование файла зависимостей
COPY requirements.txt .

# Установка Python зависимостей с кэшированием
RUN pip install --no-cache-dir --user -r requirements.txt

# Копирование исходного кода
COPY . .

# Создание директории для выходных файлов
RUN mkdir -p /app/output && chown appuser:appuser /app/output

# Открытие порта
EXPOSE 8000

# Команда запуска
CMD ["python", "main.py"]