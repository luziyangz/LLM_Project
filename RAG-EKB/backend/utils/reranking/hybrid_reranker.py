from typing import List, Optional, Dict, Any
import numpy as np
from .base_reranker import BaseReranker
from .semantic_reranker import SemanticReranker
from .cross_encoder_reranker import CrossEncoderReranker
from ..retrieval.base_retriever import RetrievalResult

class HybridReranker(BaseReranker):
    """混合重排器 - 结合多种重排策略"""
    
    def __init__(self, model_config: dict, weights: Dict[str, float] = None):
        self.weights = weights or {
            'semantic': 0.4,
            'cross_encoder': 0.4, 
            'original': 0.2
        }
        
        # 初始化子重排器
        self.semantic_reranker = SemanticReranker(model_config)
        self.cross_encoder_reranker = CrossEncoderReranker()
    
    def rerank(self, query: str, candidates: List[RetrievalResult], 
              top_k: Optional[int] = None) -> List[RetrievalResult]:
        """混合重排策略"""
        if not candidates:
            return candidates
        
        if top_k is None:
            top_k = len(candidates)
        
        # 保存原始分数
        original_scores = [result.score for result in candidates]
        
        try:
            # 语义重排
            semantic_results = self.semantic_reranker.rerank(query, candidates.copy())
            semantic_scores = [r.metadata.get('semantic_score', 0) for r in semantic_results]
            
            # 交叉编码器重排
            cross_results = self.cross_encoder_reranker.rerank(query, candidates.copy())
            cross_scores = [r.metadata.get('cross_encoder_score', 0) for r in cross_results]
            
            # 分数归一化
            semantic_scores = self._normalize_scores(semantic_scores)
            cross_scores = self._normalize_scores(cross_scores)
            original_scores = self._normalize_scores(original_scores)
            
            # 加权融合
            for i, result in enumerate(candidates):
                combined_score = (
                    self.weights['semantic'] * semantic_scores[i] +
                    self.weights['cross_encoder'] * cross_scores[i] +
                    self.weights['original'] * original_scores[i]
                )
                result.score = combined_score
                result.metadata.update({
                    'semantic_score': semantic_scores[i],
                    'cross_encoder_score': cross_scores[i],
                    'original_score': original_scores[i],
                    'hybrid_score': combined_score
                })
            
            # 最终排序
            reranked = sorted(candidates, key=lambda x: x.score, reverse=True)
            return reranked[:top_k]
            
        except Exception as e:
            print(f"混合重排失败: {e}")
            return candidates[:top_k]
    
    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """分数归一化"""
        if not scores:
            return scores
        
        scores = np.array(scores)
        min_score = scores.min()
        max_score = scores.max()
        
        if max_score == min_score:
            return [0.5] * len(scores)
        
        normalized = (scores - min_score) / (max_score - min_score)
        return normalized.tolist()