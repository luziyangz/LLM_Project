from sentence_transformers import SentenceTransformer
from openai import OpenAI
import os
import logging  # 添加这行
from utils.logs_utils import LoggerConfig, log_decorator
import requests
from huggingface_hub import snapshot_download
import torch

logger = LoggerConfig().get_logger()

class EmbeddingModelLoader:
    """
    嵌入模型加载器
    支持：
    1. 本地模型加载
    2. Hugging Face模型下载和加载
    3. OpenAI API调用
    """
    
    def __init__(self, model_config):
        self.model_config = model_config
        self.model = None
        
    @log_decorator(level=logging.INFO)
    def load_model(self):
        """加载模型的主入口"""
        try:
            if self.model_config.get("type") == "local":
                self.model = self._load_local_model()
            elif self.model_config.get("type") == "huggingface":
                self.model = self._load_huggingface_model()
            else:
                self.model = self._load_api_model()
            return self.model
        except Exception as e:
            logger.error(f"模型加载失败: {str(e)}", exc_info=True)
            raise
            
    @log_decorator(level=logging.DEBUG)
    def _load_local_model(self):
        """加载本地模型"""
        model_path = self.model_config.get("path")
        if not os.path.exists(model_path):
            logger.warning(f"本地模型路径不存在: {model_path}")
            return self._load_huggingface_model()
            
        try:
            model = SentenceTransformer(model_path)
            logger.info(f"成功加载本地模型: {model_path}")
            return model
        except Exception as e:
            logger.error(f"本地模型加载失败: {str(e)}")
            return self._load_huggingface_model()
            
    @log_decorator(level=logging.DEBUG)
    def _load_huggingface_model(self):
        """从Hugging Face下载并加载模型"""
        model_name = self.model_config.get("huggingface_model")
        # 修改默认缓存目录为项目指定路径
        cache_dir = self.model_config.get("cache_dir", "E:/code/AIProjectCode/trae_code/project/RAG-EKB/backend/utils/models")
        
        try:
            # 确保缓存目录存在
            os.makedirs(cache_dir, exist_ok=True)
            
            # 检查本地是否已存在模型
            local_model_path = os.path.join(cache_dir, model_name.split('/')[-1])
            if os.path.exists(local_model_path):
                logger.info(f"发现本地缓存模型: {local_model_path}")
                model = SentenceTransformer(local_model_path)
                return model
            
            # 从Hugging Face下载模型
            logger.info(f"开始从Hugging Face下载模型: {model_name}")
            model_path = snapshot_download(
                repo_id=model_name,
                cache_dir=cache_dir,
                local_files_only=False
            )
            
            # 加载模型并保存到指定路径
            model = SentenceTransformer(model_path)
            model.save(local_model_path)
            logger.info(f"模型已保存到本地路径: {local_model_path}")
            
            return model
        except Exception as e:
            logger.error(f"Hugging Face模型加载失败: {str(e)}")
            return self._load_api_model()
            
    @log_decorator(level=logging.DEBUG)
    def _load_api_model(self):
        """配置API调用，支持多种模型服务提供商"""
        try:
            provider = self.model_config.get("provider", "openai")  # 默认为openai
            api_key = self.model_config.get("api_key")
            base_url = self.model_config.get("api_base")
            model_name = self.model_config.get("model_name")
            
            if not api_key:
                raise ValueError("未提供API密钥")
                
            # 根据不同的提供商配置客户端
            if provider == "aliyun":
                if not base_url:
                    base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
                logger.info("使用阿里云百炼模型服务")
            elif provider == "baidu":
                if not base_url:
                    base_url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/embeddings"
                logger.info("使用百度文心大模型服务")
            elif provider == "tencent":
                if not base_url:
                    base_url = "https://hunyuan.cloud.tencent.com/hyllm/v1"
                logger.info("使用腾讯混元大模型服务")
            else:
                logger.info("使用OpenAI模型服务")
            
            client = OpenAI(
                api_key=api_key,
                base_url=base_url
            )
            logger.info(f"成功配置{provider}模型服务")
            return client.embeddings
            
        except Exception as e:
            logger.error(f"API模型配置失败: {str(e)}")
            raise