import asyncio
import asyncpg
import os

async def init_database():
    """Инициализация базы данных"""
    # Получаем параметры подключения из Railway
    database_url = os.getenv('DATABASE_URL')
    
    if database_url:
        conn = await asyncpg.connect(database_url)
    else:
        conn = await asyncpg.connect(
            host=os.getenv('DB_HOST'),
            port=int(os.getenv('DB_PORT', 5432)),
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASSWORD'),
            database=os.getenv('DB_NAME')
        )
    
    # Создание таблицы samples
    await conn.execute('''
        CREATE TABLE IF NOT EXISTS samples (
            id SERIAL PRIMARY KEY,
            sample_type VARCHAR(50) NOT NULL,
            sample_name VARCHAR(100) NOT NULL,
            sample_bpm FLOAT NOT NULL,
            sample_beats INTEGER NOT NULL,
            sample_pattern FLOAT[] NOT NULL,
            sample_cid VARCHAR(100) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    ''')
    
    # Создание индексов
    await conn.execute('''
        CREATE INDEX IF NOT EXISTS idx_samples_type_name 
        ON samples(sample_type, sample_name);
    ''')
    
    print("База данных инициализирована успешно")
    await conn.close()

if __name__ == "__main__":
    asyncio.run(init_database())