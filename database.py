import asyncpg
from typing import List, Dict, Any
from config import db_config

class DatabaseManager:
    """Менеджер для работы с базой данных"""
    
    def __init__(self):
        self.pool = None
    
    async def create_pool(self):
        """Создает пул соединений с базой данных"""
        try:
                       
            print("Подключение к БД через DATABASE_URL")
            self.pool = await asyncpg.create_pool(
                db_config.database_url,
                min_size=1,
                max_size=5,  
                command_timeout=60
            )            
            print("Пул соединений с базой данных создан успешно")
        except Exception as e:
            print(f"Ошибка создания пула соединений: {e}")
            raise
    
    async def close_pool(self):
        """Закрывает пул соединений"""
        if self.pool:
            await self.pool.close()
            print("Пул соединений закрыт")
    
    async def fetch_samples(self) -> List[Dict[str, Any]]:
        """Получает все сэмплы из базы данных"""
        if not self.pool:
            await self.create_pool()
        
        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT sample_type, sample_name, sample_bpm, sample_beats, sample_pattern, sample_cid
                    FROM samples
                    ORDER BY sample_type, sample_name
                """)
                
                return [dict(row) for row in rows]
        except Exception as e:
            print(f"Ошибка получения сэмплов: {e}")
            return []
    
    async def get_sample_by_traits(self, trait_type: str, sample_name: str) -> Dict[str, Any]:
        """Получает конкретный сэмпл по типу и имени"""
        if not self.pool:
            await self.create_pool()
        
        try:
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow("""
                    SELECT sample_type, sample_name, sample_bpm, sample_beats, sample_pattern, sample_cid
                    FROM samples
                    WHERE sample_type = $1 AND sample_name = $2
                """, trait_type, sample_name)
                
                return dict(row) if row else None
        except Exception as e:
            print(f"Ошибка получения сэмпла {trait_type}/{sample_name}: {e}")
            return None
    
    async def health_check(self) -> bool:
        """Проверка состояния подключения к базе данных"""
        try:
            if not self.pool:
                await self.create_pool()
            
            async with self.pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
                return True
        except Exception as e:
            print(f"Ошибка проверки БД: {e}")
            return False

# Глобальный экземпляр менеджера базы данных
db_manager = DatabaseManager()