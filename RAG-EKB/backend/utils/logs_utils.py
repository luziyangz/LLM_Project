import logging
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
import os
from typing import Optional
from functools import wraps
import time
import traceback

class LoggerConfig:
    """
    企业级日志配置类
    功能：
    1. 支持文件日志和控制台日志
    2. 支持日志级别控制
    3. 支持日志文件大小轮转
    4. 支持日志文件按时间轮转
    5. 支持日志装饰器，用于函数调用追踪
    6. 支持异常堆栈跟踪
    """
    
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        """单例模式"""
        if not cls._instance:
            cls._instance = super(LoggerConfig, cls).__new__(cls)
        return cls._instance
    
    def __init__(
        self,
        log_dir: str = "E:/code/AIProjectCode/trae_codeproject/RAG-EKB/backend/log",
        log_level: int = logging.INFO,
        enable_console: bool = True,
        log_format: Optional[str] = None,
        max_bytes: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5,
        encoding: str = 'utf-8',
        tag: str = 'DEFAULT'  # 添加tag参数
    ):
        if hasattr(self, '_initialized'):
            return
            
        self.log_dir = log_dir
        self.log_level = log_level
        self.enable_console = enable_console
        self.log_format = log_format or '%(asctime)s - [%(tag)s] - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        self.encoding = encoding
        self.tag = tag  # 保存tag
        
        # 创建日志目录
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        self._initialized = True
            
    def get_logger(self, name: str = 'rag_backend') -> logging.Logger:
        """
        获取logger实例
        :param name: logger名称
        :return: logger实例
        """
        # 创建logger
        logger = logging.getLogger(name)
        logger.setLevel(self.log_level)
        
        # 如果logger已经有handler，先清除
        if logger.handlers:
            logger.handlers.clear()
            
        # 创建一个过滤器来添加tag
        class TagFilter(logging.Filter):
            def __init__(self, tag):
                super().__init__()
                self.tag = tag
                
            def filter(self, record):
                record.tag = self.tag
                return True
                
        # 添加tag过滤器
        tag_filter = TagFilter(self.tag)
        logger.addFilter(tag_filter)
            
        # 创建格式化器
        formatter = logging.Formatter(self.log_format)
        
        # 创建按大小轮转的文件处理器
        size_handler = RotatingFileHandler(
            filename=os.path.join(self.log_dir, f'{name}.log'),
            maxBytes=self.max_bytes,
            backupCount=self.backup_count,
            encoding=self.encoding
        )
        size_handler.setFormatter(formatter)
        logger.addHandler(size_handler)
        
        # 创建按时间轮转的文件处理器（每天）
        time_handler = TimedRotatingFileHandler(
            filename=os.path.join(self.log_dir, f'{name}_time.log'),
            when='midnight',
            interval=1,
            backupCount=self.backup_count,
            encoding=self.encoding
        )
        time_handler.setFormatter(formatter)
        logger.addHandler(time_handler)
        
        # 如果启用控制台输出
        if self.enable_console:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
            
        return logger

def log_decorator(level=logging.INFO):
    """
    日志装饰器，用于记录函数调用
    :param level: 日志级别
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = LoggerConfig().get_logger()
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                end_time = time.time()
                logger.log(level, f"函数 {func.__name__} 执行成功，耗时: {end_time - start_time:.2f}秒")
                return result
            except Exception as e:
                end_time = time.time()
                logger.error(f"函数 {func.__name__} 执行失败，耗时: {end_time - start_time:.2f}秒")
                logger.error(f"异常信息: {str(e)}")
                logger.error(f"堆栈跟踪:\n{traceback.format_exc()}")
                raise
        return wrapper
    return decorator

# 创建默认logger实例
default_logger = LoggerConfig().get_logger()

# 导出常用的日志方法
debug = default_logger.debug
info = default_logger.info
warning = default_logger.warning
error = default_logger.error
critical = default_logger.critical

# 使用示例
if __name__ == '__main__':
    # 基本日志使用
    info("这是一条信息日志")
    error("这是一条错误日志")
    
    # 装饰器使用示例
    @log_decorator(level=logging.INFO)
    def example_function(x, y):
        return x + y
    
    try:
        result = example_function(1, 2)
    except Exception as e:
        error("函数调用失败", exc_info=True)