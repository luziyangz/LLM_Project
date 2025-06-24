from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

@dataclass
class RetrievalResult:
    """检索结果"""
    chunk_id: str
    content: str
    score: float
    metadata: Dict[str, Any]
    source_file: Optional[str] = None

class BaseRetriever(ABC):
    """向量检索器基类"""
    
    @abstractmethod
    def add_chunks(self, chunks: List[Dict[str, Any]]) -> bool:
        """添加文本块到向量库"""
        pass
    
    @abstractmethod
    def search(self, query: str, top_k: int = 5, 
              filters: Optional[Dict] = None) -> List[RetrievalResult]:
        """向量检索"""
        pass
    
    @abstractmethod
    def delete_chunks(self, chunk_ids: List[str]) -> bool:
        """删除指定的文本块"""
        pass
    
    @abstractmethod
    def update_chunk(self, chunk_id: str, new_content: str, 
                    new_metadata: Dict[str, Any]) -> bool:
        """更新文本块"""
        pass

class RetrievalManager:
    """检索管理器"""
    
    def __init__(self):
        self._retrievers: Dict[str, BaseRetriever] = {}
        self._default_retriever: Optional[str] = None
    
    def register_retriever(self, name: str, retriever: BaseRetriever, 
                          set_as_default: bool = False):
        """注册检索器"""
        self._retrievers[name] = retriever
        if set_as_default or not self._default_retriever:
            self._default_retriever = name
    
    def get_retriever(self, name: Optional[str] = None) -> BaseRetriever:
        """获取检索器"""
        retriever_name = name or self._default_retriever
        if not retriever_name or retriever_name not in self._retrievers:
            raise ValueError(f"检索器 {retriever_name} 不存在")
        return self._retrievers[retriever_name]
    
    def search(self, query: str, retriever_name: Optional[str] = None, 
              **kwargs) -> List[RetrievalResult]:
        """执行检索"""
        retriever = self.get_retriever(retriever_name)
        return retriever.search(query, **kwargs)