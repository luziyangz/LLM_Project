import os
from typing import Dict, Any

class EmbeddingConfig:
    """嵌入模型配置管理"""
    
    @staticmethod
    def get_aliyun_config() -> Dict[str, Any]:
        """阿里云百炼嵌入配置"""
        return {
            "type": "api",
            "provider": "aliyun",
            "api_key": os.getenv("DASHSCOPE_API_KEY"),
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "model_name": "text-embedding-v1"
        }
    
    @staticmethod
    def get_openai_config() -> Dict[str, Any]:
        """OpenAI嵌入配置"""
        return {
            "type": "api",
            "provider": "openai",
            "api_key": os.getenv("OPENAI_API_KEY"),
            "base_url": "https://api.openai.com/v1",
            "model_name": "text-embedding-ada-002"
        }
    
    @staticmethod
    def get_local_config() -> Dict[str, Any]:
        """本地模型配置"""
        return {
            "type": "local",
            "path": "E:/code/AIProjectCode/trae_code/project/RAG-EKB/backend/utils/models/all-MiniLM-L6-v2",
            "huggingface_model": "sentence-transformers/all-MiniLM-L6-v2",
            "cache_dir": "E:/code/AIProjectCode/trae_code/project/RAG-EKB/backend/utils/models"
        }