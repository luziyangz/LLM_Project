from sqlalchemy import create_engine, inspect
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from contextlib import asynccontextmanager
from utils.logs_utils import LoggerConfig, log_decorator
import logging
import asyncio
from sqlalchemy import text

logger = LoggerConfig().get_logger()

class DatabaseManager:
    def __init__(self, connection_string):
        self.connection_string = connection_string
        self.engine = None
        self.SessionLocal = None
        
    @log_decorator(level=logging.INFO)
    def init_db(self):
        try:
            # 优化的连接池配置
            self.engine = create_async_engine(
                self.connection_string,
                pool_size=10,           # 增加连接池大小
                max_overflow=20,        # 增加最大溢出连接数
                pool_timeout=60,        # 增加连接超时时间
                pool_recycle=3600,      # 设置连接回收时间（1小时）
                pool_pre_ping=True,     # 启用连接预检查
                connect_args={
                    "check_same_thread": False,  # SQLite特定配置
                    "timeout": 30                # SQLite连接超时
                }
            )
            
            self.SessionLocal = sessionmaker(
                autocommit=False, 
                autoflush=False, 
                bind=self.engine,
                class_=AsyncSession
            )
            
            return self.engine, self.SessionLocal
            
        except Exception as e:
            logger.error(f"数据库初始化失败: {str(e)}", exc_info=True)
            raise
    
    async def ensure_connection(self):
        """确保数据库连接有效"""
        try:
            async with self.SessionLocal() as session:
                await session.execute(text("SELECT 1"))
        except Exception as e:
            logger.warning(f"数据库连接检查失败，尝试重新初始化: {e}")
            self.init_db()
    
    @asynccontextmanager
    async def get_session(self):
        """获取数据库会话 - 异步上下文管理器"""
        if self.SessionLocal is None:
            raise RuntimeError("数据库未初始化，请先调用 init_db()")
        
        session = self.SessionLocal()
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
    
    @asynccontextmanager
    async def get_robust_session(self):
        """获取健壮的数据库会话"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                await self.ensure_connection()
                session = self.SessionLocal()
                yield session
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                logger.warning(f"数据库会话创建失败，重试 {attempt + 1}/{max_retries}: {e}")
                await asyncio.sleep(1)
            finally:
                if 'session' in locals():
                    await session.close()
            
    def check_and_create_tables(self, Base):
        try:
            async def _check_and_create_tables():
                async with self.engine.begin() as conn:
                    await conn.run_sync(lambda sync_conn: Base.metadata.create_all(sync_conn))
                    logger.info("所有表已通过 Base.metadata.create_all 检查和创建（如果不存在）。")
            
            import asyncio
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(_check_and_create_tables())
            else:
                loop.run_until_complete(_check_and_create_tables())
                    
        except Exception as e:
            logger.error(f"检查和创建表失败: {str(e)}", exc_info=True)
