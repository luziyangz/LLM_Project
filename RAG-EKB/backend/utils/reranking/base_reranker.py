from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from ..retrieval.base_retriever import RetrievalResult

class BaseReranker(ABC):
    """重排器基类"""
    
    @abstractmethod
    def rerank(self, query: str, candidates: List[RetrievalResult], 
              top_k: Optional[int] = None) -> List[RetrievalResult]:
        """重排候选结果"""
        pass

class RerankingManager:
    """重排管理器"""
    
    def __init__(self):
        self._rerankers: Dict[str, BaseReranker] = {}
        self._default_reranker: Optional[str] = None
    
    def register_reranker(self, name: str, reranker: BaseReranker, 
                         set_as_default: bool = False):
        """注册重排器"""
        self._rerankers[name] = reranker
        if set_as_default or not self._default_reranker:
            self._default_reranker = name
    
    def rerank(self, query: str, candidates: List[RetrievalResult], 
              reranker_name: Optional[str] = None, **kwargs) -> List[RetrievalResult]:
        """执行重排"""
        if not self._rerankers:
            return candidates  # 如果没有重排器，直接返回原结果
        
        reranker_name = reranker_name or self._default_reranker
        if reranker_name and reranker_name in self._rerankers:
            reranker = self._rerankers[reranker_name]
            return reranker.rerank(query, candidates, **kwargs)
        
        return candidates