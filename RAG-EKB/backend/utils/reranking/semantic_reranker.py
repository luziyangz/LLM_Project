from typing import List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from .base_reranker import BaseReranker
from ..retrieval.base_retriever import RetrievalResult
from ..embedding_utils import EmbeddingModelLoader

class SemanticReranker(BaseReranker):
    """基于语义相似度的重排器"""
    
    def __init__(self, model_config: dict):
        self.model_config = model_config
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """加载嵌入模型"""
        try:
            loader = EmbeddingModelLoader(self.model_config)
            self.model = loader.load_model()
        except Exception as e:
            print(f"模型加载失败: {e}")
            # 使用默认模型
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def rerank(self, query: str, candidates: List[RetrievalResult], 
              top_k: Optional[int] = None) -> List[RetrievalResult]:
        """基于语义相似度重排"""
        if not candidates:
            return candidates
        
        if top_k is None:
            top_k = len(candidates)
        
        try:
            # 获取查询和候选文档的嵌入
            query_embedding = self.model.encode([query])
            candidate_texts = [result.content for result in candidates]
            candidate_embeddings = self.model.encode(candidate_texts)
            
            # 计算相似度
            similarities = cosine_similarity(query_embedding, candidate_embeddings)[0]
            
            # 更新分数并排序
            for i, result in enumerate(candidates):
                # 结合原始分数和语义相似度
                semantic_score = float(similarities[i])
                combined_score = 0.7 * semantic_score + 0.3 * result.score
                result.score = combined_score
                result.metadata['semantic_score'] = semantic_score
                result.metadata['original_score'] = result.score
            
            # 按新分数排序
            reranked = sorted(candidates, key=lambda x: x.score, reverse=True)
            return reranked[:top_k]
            
        except Exception as e:
            print(f"语义重排失败: {e}")
            return candidates[:top_k]