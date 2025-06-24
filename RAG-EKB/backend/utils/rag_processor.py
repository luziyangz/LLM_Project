from typing import List, Dict, Any, Optional
from .chunkers.chunk_manager import ChunkManager
from .retrieval.base_retriever import RetrievalResult, RetrievalManager
from .reranking.base_reranker import RerankingManager
from .reranking import SemanticReranker, CrossEncoderReranker, HybridReranker, RerankConfig

class RAGProcessor:
    """RAG处理器 - 整合分块、检索、重排功能"""
    
    def __init__(self):
        self.chunk_manager = ChunkManager()
        self.retrieval_manager = RetrievalManager()
        self.reranking_manager = RerankingManager()
        self._setup_rerankers()
    
    def _setup_rerankers(self):
        """设置重排器"""
        # 语义重排器
        semantic_config = RerankConfig.get_semantic_config()
        semantic_reranker = SemanticReranker(semantic_config)
        self.reranking_manager.register_reranker(
            "semantic", semantic_reranker, set_as_default=True
        )
        
        # 交叉编码器重排器
        cross_encoder_reranker = CrossEncoderReranker()
        self.reranking_manager.register_reranker(
            "cross_encoder", cross_encoder_reranker
        )
        
        # 混合重排器
        hybrid_config = RerankConfig.get_hybrid_config()
        hybrid_reranker = HybridReranker(hybrid_config["model_config"], hybrid_config["weights"])
        self.reranking_manager.register_reranker(
            "hybrid", hybrid_reranker
        )
    
    def process_documents(self, documents: List[Dict[str, str]], 
                         chunker_type: str = 'fixed_length', 
                         **chunker_kwargs) -> Dict[str, Any]:
        """处理文档：分块"""
        # 分块
        chunk_results = self.chunk_manager.chunk_multiple_texts(
            documents, chunker_type, **chunker_kwargs
        )
        
        # 统计信息
        total_chunks = sum(len(chunks) for chunks in chunk_results.values())
        
        return {
            'chunks': chunk_results,
            'total_documents': len(documents),
            'total_chunks': total_chunks,
            'chunker_type': chunker_type,
            'chunker_config': chunker_kwargs
        }
    
    def search_and_rerank(self, query: str, top_k: int = 10, 
                         rerank_top_k: Optional[int] = None,
                         retriever_name: Optional[str] = None,
                         reranker_name: Optional[str] = None) -> List[RetrievalResult]:
        """检索和重排"""
        # 检索阶段
        results = self.retrieval_manager.search(
            query, retriever_name, top_k=top_k
        )
        
        if not results:
            return []
        
        # 重排阶段
        reranked_results = self.reranking_manager.rerank(
            query, results, reranker_name, top_k=rerank_top_k
        )
        
        return reranked_results