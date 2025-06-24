from typing import List, Optional
import torch
from sentence_transformers import CrossEncoder

from .base_reranker import BaseReranker
from ..retrieval.base_retriever import RetrievalResult

class CrossEncoderReranker(BaseReranker):
    """基于交叉编码器的重排器"""
    
    def __init__(self, model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2'):
        self.model_name = model_name
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """加载交叉编码器模型"""
        try:
            self.model = CrossEncoder(self.model_name)
        except Exception as e:
            print(f"交叉编码器加载失败: {e}")
            # 降级到默认模型
            try:
                self.model = CrossEncoder('cross-encoder/ms-marco-TinyBERT-L-2-v2')
            except:
                self.model = None
    
    def rerank(self, query: str, candidates: List[RetrievalResult], 
              top_k: Optional[int] = None) -> List[RetrievalResult]:
        """使用交叉编码器重排"""
        if not candidates or self.model is None:
            return candidates[:top_k] if top_k else candidates
        
        if top_k is None:
            top_k = len(candidates)
        
        try:
            # 构建查询-文档对
            query_doc_pairs = [(query, result.content) for result in candidates]
            
            # 批量预测相关性分数
            scores = self.model.predict(query_doc_pairs)
            
            # 更新结果分数
            for i, result in enumerate(candidates):
                cross_encoder_score = float(scores[i])
                # 结合原始分数和交叉编码器分数
                combined_score = 0.8 * cross_encoder_score + 0.2 * result.score
                result.score = combined_score
                result.metadata['cross_encoder_score'] = cross_encoder_score
                result.metadata['original_score'] = result.score
            
            # 按新分数排序
            reranked = sorted(candidates, key=lambda x: x.score, reverse=True)
            return reranked[:top_k]
            
        except Exception as e:
            print(f"交叉编码器重排失败: {e}")
            return candidates[:top_k]