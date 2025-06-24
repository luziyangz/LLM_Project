from typing import Dict, Any
import os

class RerankConfig:
    """重排器配置管理"""
    
    @staticmethod
    def get_semantic_config() -> Dict[str, Any]:
        """语义重排器配置"""
        return {
            "type": "local",
            "path": "E:/code/AIProjectCode/trae_code/project/RAG-EKB/backend/utils/models/all-MiniLM-L6-v2",
            "huggingface_model": "sentence-transformers/all-MiniLM-L6-v2",
            "cache_dir": "E:/code/AIProjectCode/trae_code/project/RAG-EKB/backend/utils/models"
        }
    
    @staticmethod
    def get_cross_encoder_config() -> Dict[str, Any]:
        """交叉编码器配置"""
        return {
            "model_name": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "fallback_model": "cross-encoder/ms-marco-TinyBERT-L-2-v2"
        }
    
    @staticmethod
    def get_hybrid_config() -> Dict[str, Any]:
        """混合重排器配置"""
        return {
            "weights": {
                "semantic": 0.4,
                "cross_encoder": 0.4,
                "original": 0.2
            },
            "model_config": RerankConfig.get_semantic_config()
        }