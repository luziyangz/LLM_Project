from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker
from utils.logs_utils import LoggerConfig, log_decorator
import logging

logger = LoggerConfig().get_logger()

class DatabaseManager:
    def __init__(self, connection_string):
        self.connection_string = connection_string
        self.engine = None
        self.SessionLocal = None
        
    @log_decorator(level=logging.INFO)
    def init_db(self):
        try:
            self.engine = create_engine(
                self.connection_string,
                pool_size=5,
                max_overflow=10,
                pool_timeout=30
            )
            
            self.SessionLocal = sessionmaker(
                autocommit=False, 
                autoflush=False, 
                bind=self.engine
            )
            
            return self.engine, self.SessionLocal
            
        except Exception as e:
            logger.error(f"数据库初始化失败: {str(e)}", exc_info=True)
            raise
            
    def check_and_create_tables(self, Base):
        try:
            inspector = inspect(self.engine)
            existing_tables = inspector.get_table_names()
            
            for table_name in Base.metadata.tables.keys():
                if table_name not in existing_tables:
                    Base.metadata.tables[table_name].create(bind=self.engine)
                    logger.info(f"创建表 {table_name} 成功")
                else:
                    logger.info(f"表 {table_name} 已存在，跳过创建")
                    
        except Exception as e:
            logger.error(f"检查和创建表失败: {str(e)}", exc_info=True)
            raise