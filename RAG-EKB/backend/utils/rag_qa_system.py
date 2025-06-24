import numpy as np
import os
import asyncio
from typing import List, Dict, Any, Optional
from .document_processor import DocumentProcessor
import faiss
from datetime import datetime
from utils.logs_utils import LoggerConfig, log_decorator
from utils.embedding_utils import EmbeddingModelLoader
from openai import OpenAI
import os
import ssl
from .chunkers.chunk_manager import ChunkManager
# 新增重排相关导入
from .rag_processor import RAGProcessor
from .reranking import SemanticReranker, CrossEncoderReranker, HybridReranker, RerankConfig
from .retrieval.base_retriever import RetrievalResult

class RAGQASystem:
    def __init__(self, embedding_config: Dict[str, Any], 
                 index_dimension: int = 768,
                 chunker_type: str = 'fixed_length',
                 chunker_config: Dict[str, Any] = None,
                 enable_rerank: bool = True,
                 rerank_strategy: str = 'hybrid'):
        """初始化RAG问答系统"""
        self.logger = LoggerConfig().get_logger()
        self.faiss_index = None
        self.metadata = []
        
        # 配置嵌入模型
        self.embedding_config = embedding_config or {
            "type": "api",
            "provider": "aliyun",
            "api_key": os.getenv("DASHSCOPE_API_KEY"),
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "model_name": "text-embedding-v1"
        }
        
        # 初始化嵌入模型
        self._init_embedding_model()
        
        # 初始化分块管理器
        self.chunk_manager = ChunkManager()
        self.chunker_type = chunker_type
        self.chunker_config = chunker_config or {}
        
        # 新增：初始化重排功能
        self.enable_rerank = enable_rerank
        self.rerank_strategy = rerank_strategy
        if self.enable_rerank:
            self._init_rerankers()
        
        self.logger.info(f"RAG系统初始化完成，使用分块器: {chunker_type}, 重排策略: {rerank_strategy if enable_rerank else '禁用'}")
    
    def _init_rerankers(self):
        """初始化重排器"""
        try:
            # 语义重排器
            semantic_config = RerankConfig.get_semantic_config()
            self.semantic_reranker = SemanticReranker(semantic_config)
            
            # 交叉编码器重排器
            self.cross_encoder_reranker = CrossEncoderReranker()
            
            # 混合重排器
            hybrid_config = RerankConfig.get_hybrid_config()
            self.hybrid_reranker = HybridReranker(
                hybrid_config["model_config"], 
                hybrid_config["weights"]
            )
            
            self.logger.info("重排器初始化完成")
        except Exception as e:
            self.logger.error(f"重排器初始化失败: {e}")
            self.enable_rerank = False
    
    async def add_documents(self, documents: List[Dict[str, str]],
                      chunker_type: str = None) -> bool:
        """添加文档到索引"""
        try:
            self.logger.info(f"开始添加 {len(documents)} 个文档到索引")
            
            # 使用指定的分块器或默认分块器
            current_chunker_type = chunker_type or self.chunker_type
            self.logger.info(f"使用分块器类型: {current_chunker_type}")
            
            all_chunks = []
            for i, doc in enumerate(documents):
                content = doc.get('content', '')
                source_file = doc.get('source_file', 'unknown')
                
                self.logger.info(f"处理文档 {i+1}/{len(documents)}: {source_file}, 内容长度: {len(content)}")
                
                # 使用分块管理器进行文本分块
                chunks = self.chunk_manager.chunk_text(
                    text=content,
                    chunker_type=current_chunker_type,
                    source_file=source_file,
                    **self.chunker_config
                )
                
                self.logger.info(f"文档 {source_file} 分块完成，生成 {len(chunks)} 个分块")
                all_chunks.extend(chunks)
            
            self.logger.info(f"总共生成 {len(all_chunks)} 个文档分块")
            
            # 生成嵌入向量
            successful_chunks = 0
            for i, chunk in enumerate(all_chunks):
                try:
                    self.logger.debug(f"处理分块 {i+1}/{len(all_chunks)}: {chunk.metadata.source_file}")
                    
                    embedding = await self._generate_embeddings(chunk.content)
                    if embedding is not None:
                        # 添加到FAISS索引
                        self._add_to_index(
                            embedding=embedding,
                            content=chunk.content,
                            metadata={
                                'chunk_id': chunk.metadata.chunk_id,
                                'source_file': chunk.metadata.source_file,
                                'start_pos': chunk.metadata.start_pos,
                                'end_pos': chunk.metadata.end_pos,
                                'chunk_type': chunk.metadata.chunk_type.value,
                                'char_count': chunk.metadata.char_count,
                                'word_count': chunk.metadata.word_count
                            }
                        )
                        successful_chunks += 1
                        
                        if (i + 1) % 10 == 0:  # 每10个分块记录一次进度
                            self.logger.info(f"已处理 {i+1}/{len(all_chunks)} 个分块")
                    else:
                        self.logger.warning(f"分块 {i+1} 嵌入生成失败")
                except Exception as e:
                    self.logger.error(f"处理分块 {i+1} 时出错: {e}")
                    continue
            
            self.logger.info(f"成功处理 {successful_chunks}/{len(all_chunks)} 个分块")
            
            # 添加自动保存机制
            try:
                index_path = "./data/faiss_index/index.faiss"
                os.makedirs(os.path.dirname(index_path), exist_ok=True)
                self.save_index(index_path)
                self.logger.info(f"索引已自动保存到: {index_path}")
            except Exception as e:
                self.logger.error(f"自动保存索引失败: {e}")
            
            # 记录最终统计信息
            stats = self.get_stats()
            self.logger.info(f"添加完成 - 总文档数: {stats['total_documents']}, 总分块数: {stats['total_chunks']}, 索引大小: {stats['index_size']}")
            
            return successful_chunks > 0
            
        except Exception as e:
            self.logger.error(f"添加文档失败: {e}", exc_info=True)
            return False
    
    async def _reload_index_if_needed(self):
        """在需要时重新加载索引"""
        try:
            index_path = "data/faiss_index/index.faiss"
            if os.path.exists(index_path):
                self.load_index(index_path)
                self.logger.info("索引重新加载成功")
        except Exception as e:
            self.logger.error(f"索引重新加载失败: {e}")
    
    def _init_embedding_model(self):
        """初始化嵌入模型"""
        try:
            if self.embedding_config.get("type") == "api":
                # 使用API模式
                self._init_api_embedding()
            else:
                # 使用本地模型
                self.embedding_loader = EmbeddingModelLoader(self.embedding_config)
                self.embedding_model = self.embedding_loader.load_model()
            
            self.logger.info("嵌入模型初始化成功")
        except Exception as e:
            self.logger.error(f"嵌入模型初始化失败: {e}")
            raise
    
    def _init_api_embedding(self):
        """初始化API嵌入模型"""
        provider = self.embedding_config.get("provider", "openai")
        api_key = self.embedding_config.get("api_key")
        base_url = self.embedding_config.get("base_url")
        
        if not api_key:
            raise ValueError("未提供API密钥")
        
        # 配置SSL上下文（用于测试环境）
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        self.api_client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        
        self.logger.info(f"使用{provider}嵌入服务")
    
    async def _generate_embeddings(self, texts):
        """生成文本嵌入"""
        try:
            if self.embedding_config.get("type") == "api":
                # 处理单个文本或文本列表
                if isinstance(texts, str):
                    embedding = await self._generate_api_embedding(texts)
                    return np.array([embedding]).astype('float32')
                else:
                    embeddings = []
                    for text in texts:
                        embedding = await self._generate_api_embedding(text)
                        embeddings.append(embedding)
                    return np.array(embeddings).astype('float32')
            else:
                # 使用本地模型
                if isinstance(texts, str):
                    texts = [texts]
                embeddings = self.embedding_model.encode(texts)
                return np.array(embeddings).astype('float32')
        except Exception as e:
            self.logger.error(f"生成嵌入失败: {e}")
            return None
    
    async def _generate_api_embedding(self, text: str) -> List[float]:
        """使用API生成单个文本的嵌入"""
        try:
            model_name = self.embedding_config.get("model_name", "text-embedding-v1")
            
            response = self.api_client.embeddings.create(
                model=model_name,
                input=text,
                timeout=30
            )
            
            return response.data[0].embedding
            
        except Exception as e:
            self.logger.error(f"API嵌入生成失败: {e}")
            raise
    
    # 修复 _add_to_index 方法
    def _add_to_index(self, embedding: np.ndarray, content: str, metadata: dict):
        """添加单个嵌入到FAISS索引"""
        try:
            # 创建索引（如果不存在）
            if self.faiss_index is None:
                dimension = len(embedding)
                self.faiss_index = faiss.IndexFlatIP(dimension)
                self.logger.info(f"创建FAISS索引，维度: {dimension}")
            
            # 添加向量
            self.faiss_index.add(np.array([embedding]).astype('float32'))
            
            # 添加元数据
            self.metadata.append({
                'content': content,
                'created_at': datetime.now().isoformat(),
                **metadata
            })
            
        except Exception as e:
            self.logger.error(f"添加到索引失败: {e}")
            raise
    
    # 修复 _generate_embeddings 方法
    async def _generate_embeddings(self, text: str) -> np.ndarray:
        """生成单个文本的嵌入"""
        try:
            if self.embedding_config.get("type") == "api":
                embedding = await self._generate_api_embedding(text)
                return np.array(embedding).astype('float32')
            else:
                embedding = self.embedding_model.encode([text])[0]
                return np.array(embedding).astype('float32')
        except Exception as e:
            self.logger.error(f"生成嵌入失败: {e}")
            return None
    
    async def retrieve_relevant_docs(self, query: str, top_k: int = 5, 
                                   rerank_top_k: Optional[int] = None,
                                   use_rerank: bool = None) -> List[Dict[str, Any]]:
        """检索相关文档（支持重排）"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                self.logger.info(f"开始检索相关文档，查询: '{query}', top_k: {top_k}, 尝试: {attempt + 1}")
                
                # 检查索引状态
                if self.faiss_index is None:
                    self.logger.warning("FAISS索引为空，尝试重新加载")
                    # 尝试重新加载索引
                    await self._reload_index_if_needed()
                    if self.faiss_index is None:
                        self.logger.warning("FAISS索引为空，无法检索")
                        return []
                
                if len(self.metadata) == 0:
                    self.logger.warning("元数据为空，无法检索")
                    return []
            
                # 确定是否使用重排
                use_rerank = use_rerank if use_rerank is not None else self.enable_rerank
                rerank_top_k = rerank_top_k or top_k
                
                # 如果启用重排，初始检索更多候选文档
                initial_top_k = top_k * 3 if use_rerank else top_k
                initial_top_k = min(initial_top_k, len(self.metadata))
                
                self.logger.info(f"索引状态: 总向量数={self.faiss_index.ntotal}, 元数据数={len(self.metadata)}")
                
                # 生成查询嵌入
                self.logger.debug("正在生成查询嵌入向量...")
                query_embedding = await self._generate_embeddings(query)
                
                if query_embedding is None:
                    self.logger.error("查询嵌入生成失败")
                    return []
                
                self.logger.debug(f"查询嵌入向量维度: {query_embedding.shape}")
                
                # FAISS搜索
                search_k = min(initial_top_k, len(self.metadata))
                self.logger.debug(f"开始FAISS搜索，搜索数量: {search_k}")
                
                scores, indices = self.faiss_index.search(query_embedding.reshape(1, -1), search_k)
                
                self.logger.debug(f"FAISS搜索完成，得分: {scores[0]}, 索引: {indices[0]}")
                
                # 构建初始结果
                initial_results = []
                for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                    if idx < len(self.metadata) and idx >= 0:
                        metadata = self.metadata[idx]
                        result = {
                            'content': metadata.get('content', ''),
                            'source_file': metadata.get('source_file', 'unknown'),
                            'chunk_id': metadata.get('chunk_id', f'chunk_{idx}'),
                            'score': float(score),
                            'rank': i + 1,
                            'metadata': metadata
                        }
                        initial_results.append(result)
                
                # 重排阶段
                if use_rerank and initial_results and len(initial_results) > 1:
                    self.logger.info(f"开始重排，策略: {self.rerank_strategy}")
                    reranked_results = await self._rerank_results(query, initial_results, rerank_top_k)
                    final_results = reranked_results[:top_k]
                    self.logger.info(f"重排完成，返回 {len(final_results)} 个文档")
                else:
                    final_results = initial_results[:top_k]
                    self.logger.info(f"跳过重排，返回 {len(final_results)} 个文档")
                
                return final_results
                    
            except Exception as e:
                self.logger.error(f"检索失败 (尝试 {attempt + 1}/{max_retries}): {e}", exc_info=True)
                if attempt == max_retries - 1:
                    return []
                await asyncio.sleep(1)  # 重试前等待
        
        return []
    
    async def _rerank_results(self, query: str, candidates: List[Dict[str, Any]], 
                            top_k: int) -> List[Dict[str, Any]]:
        """重排检索结果"""
        try:
            # 转换为RetrievalResult格式
            retrieval_results = []
            for candidate in candidates:
                result = RetrievalResult(
                    chunk_id=candidate['chunk_id'],
                    content=candidate['content'],
                    score=candidate['score'],
                    metadata=candidate.get('metadata', {}),
                    source_file=candidate.get('source_file')
                )
                retrieval_results.append(result)
            
            # 选择重排器
            if self.rerank_strategy == 'semantic':
                reranker = self.semantic_reranker
            elif self.rerank_strategy == 'cross_encoder':
                reranker = self.cross_encoder_reranker
            elif self.rerank_strategy == 'hybrid':
                reranker = self.hybrid_reranker
            else:
                self.logger.warning(f"未知重排策略: {self.rerank_strategy}，使用语义重排")
                reranker = self.semantic_reranker
            
            # 执行重排
            reranked_retrieval_results = reranker.rerank(query, retrieval_results, top_k)
            
            # 转换回原格式
            reranked_results = []
            for i, result in enumerate(reranked_retrieval_results):
                reranked_result = {
                    'content': result.content,
                    'source_file': result.source_file,
                    'chunk_id': result.chunk_id,
                    'score': result.score,
                    'rank': i + 1,
                    'metadata': result.metadata,
                    'rerank_info': {
                        'strategy': self.rerank_strategy,
                        'original_score': result.metadata.get('original_score', result.score),
                        'semantic_score': result.metadata.get('semantic_score'),
                        'cross_encoder_score': result.metadata.get('cross_encoder_score'),
                        'hybrid_score': result.metadata.get('hybrid_score')
                    }
                }
                reranked_results.append(reranked_result)
            
            return reranked_results
            
        except Exception as e:
            self.logger.error(f"重排失败: {e}")
            return candidates[:top_k]
    
    def save_index(self, index_path: str):
        """保存FAISS索引到磁盘"""
        try:
            if self.faiss_index is not None:
                os.makedirs(os.path.dirname(index_path), exist_ok=True)
                faiss.write_index(self.faiss_index, index_path)
                
                # 保存元数据
                metadata_path = index_path.replace('.faiss', '_metadata.json')
                import json
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(self.metadata, f, ensure_ascii=False, indent=2)
                
                self.logger.info(f"索引已保存到: {index_path}")
            else:
                self.logger.warning("没有索引可保存")
                
        except Exception as e:
            self.logger.error(f"保存索引失败: {e}")
            raise
    
    def load_index(self, index_path: str):
        """从磁盘加载FAISS索引"""
        try:
            if os.path.exists(index_path):
                self.faiss_index = faiss.read_index(index_path)
                
                # 加载元数据
                metadata_path = index_path.replace('.faiss', '_metadata.json')
                if os.path.exists(metadata_path):
                    import json
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        self.metadata = json.load(f)
                
                self.logger.info(f"索引已从 {index_path} 加载")
            else:
                self.logger.warning(f"索引文件不存在: {index_path}")
                
        except Exception as e:
            self.logger.error(f"加载索引失败: {e}")
            raise
    
    async def add_document_with_config(self, file_path: str, config: dict):
        """使用配置添加单个文档到知识库"""
        try:
            # 使用DocumentProcessor处理文档
            doc_processor = DocumentProcessor()
            
            # 检查文件类型并处理
            if not doc_processor.is_supported(file_path):
                raise ValueError(f"不支持的文件类型: {os.path.splitext(file_path)[1]}")
            
            # 处理文档
            doc_result = doc_processor.process_document(file_path)
            content = doc_result['content']
            
            # 使用默认分块器，不需要特殊处理
            chunker_type = config.get('chunker_type', 'fixed_length')
            
            # 构造文档格式
            documents = [{
                'content': content,
                'source_file': file_path,
                'metadata': {
                    **config,
                    'file_type': doc_result['file_type'],
                    'word_count': doc_result['word_count'],
                    'processed_at': doc_result['processed_at']
                }
            }]
            
            # 使用指定分块器
            chunker_config = config.get('chunker_config', {})
            result = await self.add_documents(
                documents, 
                chunker_type=chunker_type,
                **chunker_config
            )
            
            self.logger.info(f"成功添加文档: {file_path}, 使用分块器: {chunker_type}")
            return result
            
        except Exception as e:
            self.logger.error(f"添加文档失败 {file_path}: {e}")
    
    # 在 RAGQASystem 类中添加以下方法
    
    async def add_document(self, file_path: str, filename: str = None, config: dict = None) -> dict:
        """添加单个文档（统一接口）"""
        try:
            # 使用默认配置或传入配置
            default_config = {
                'chunker_type': self.chunker_type,
                'chunker_config': self.chunker_config
            }
            if config:
                default_config.update(config)
            
            # 调用现有的配置化处理方法
            result = await self.add_document_with_config(file_path, default_config)
            
            return {
                'success': result,
                'filename': filename or os.path.basename(file_path),
                'file_path': file_path,
                'message': '文档添加成功' if result else '文档添加失败'
            }
            
        except Exception as e:
            self.logger.error(f"添加单个文档失败: {e}")
            return {
                'success': False,
                'filename': filename or os.path.basename(file_path),
                'message': str(e)
            }
    
    async def add_multiple_documents(self, file_paths: List[str], config: dict = None) -> dict:
        """添加多个文档（批量处理）"""
        try:
            results = []
            success_count = 0
            
            for file_path in file_paths:
                result = await self.add_document(file_path, config=config)
                results.append(result)
                if result['success']:
                    success_count += 1
            
            return {
                'success': success_count > 0,
                'total_files': len(file_paths),
                'success_count': success_count,
                'failed_count': len(file_paths) - success_count,
                'results': results,
                'message': f'成功处理 {success_count}/{len(file_paths)} 个文档'
            }
            
        except Exception as e:
            self.logger.error(f"批量添加文档失败: {e}")
            return {
                'success': False,
                'message': str(e),
                'results': []
            }
    
    async def add_documents_from_content(self, documents: List[Dict[str, str]], 
                                   chunker_type: str = None) -> dict:
        """从内容添加文档（保持现有逻辑）"""
        try:
            result = await self.add_documents(documents, chunker_type)
            return {
                'success': result,
                'document_count': len(documents),
                'message': f'成功处理 {len(documents)} 个文档内容' if result else '文档内容处理失败'
            }
        except Exception as e:
            self.logger.error(f"从内容添加文档失败: {e}")
            return {
                'success': False,
                'message': str(e)
            }

    async def answer_question(self, query: str, top_k: int = 5,
                            use_rerank: bool = None,
                            rerank_strategy: str = None,
                            relevance_threshold: float = 0.15,  # 降低默认阈值
                            enable_smart_rag: bool = True) -> dict:  # 新增开关
        """回答问题（支持智能RAG调用）"""
        try:
            # 临时设置重排策略
            original_strategy = self.rerank_strategy
            if rerank_strategy:
                self.rerank_strategy = rerank_strategy
            
            # 智能相关性判断（可通过参数控制）
            if enable_smart_rag:
                try:
                    is_relevant = await self._check_query_relevance(query, relevance_threshold)
                    
                    if not is_relevant:
                        self.logger.info(f"问题与知识库不相关，建议直接调用大模型: {query[:50]}...")
                        return {
                            'answer': None,  # 表示需要直接调用大模型
                            'sources': [],
                            'has_context': False,
                            'query': query,
                            'relevance_check': 'failed',
                            'rerank_enabled': False,
                            'smart_rag_enabled': True
                        }
                except Exception as e:
                    self.logger.warning(f"相关性检查异常，继续使用RAG: {e}")
                    # 相关性检查失败时，继续使用RAG
            
            # 检索相关文档
            relevant_docs = await self.retrieve_relevant_docs(
                query, 
                top_k=top_k, 
                use_rerank=use_rerank
            )
            
            # 恢复原始策略
            self.rerank_strategy = original_strategy
            
            if not relevant_docs:
                return {
                    'answer': '抱歉，我在知识库中没有找到相关信息。',
                    'sources': [],
                    'has_context': False,
                    'query': query,
                    'rerank_enabled': use_rerank if use_rerank is not None else self.enable_rerank,
                    'smart_rag_enabled': enable_smart_rag
                }
            
            # 构建上下文时控制长度
            context_parts = []
            total_length = 0
            max_context_length = 12000  # 设置上下文最大长度

            for doc in relevant_docs:
                doc_content = doc['content']
                if total_length + len(doc_content) > max_context_length:
                    # 截断最后一个文档
                    remaining_length = max_context_length - total_length
                    if remaining_length > 100:  # 至少保留100字符
                        doc_content = doc_content[:remaining_length] + "\n[文档内容已截断...]"
                        context_parts.append(doc_content)
                    break
                
                context_parts.append(doc_content)
                total_length += len(doc_content)

            context = '\n\n'.join(context_parts)
            sources = [{
                'content': doc['content'][:200] + '...' if len(doc['content']) > 200 else doc['content'],
                'score': doc['score'],
                'chunk_id': doc['chunk_id'],
                'document_id': doc.get('document_id', 'unknown'),
                'source_file': doc.get('source_file', 'unknown'),
                'rank': doc.get('rank', 0),
                'rerank_info': doc.get('rerank_info', {})
            } for doc in relevant_docs]
            
            return {
                'context': context,
                'sources': sources,
                'has_context': True,
                'query': query,
                'retrieved_count': len(relevant_docs),
                'rerank_enabled': use_rerank if use_rerank is not None else self.enable_rerank,
                'rerank_strategy': rerank_strategy or self.rerank_strategy,
                'smart_rag_enabled': enable_smart_rag,
                'relevance_check': 'passed' if enable_smart_rag else 'disabled'
            }
            
        except Exception as e:
            self.logger.error(f"回答问题失败: {e}")
            return {
                'answer': f'处理问题时出现错误: {str(e)}',
                'sources': [],
                'has_context': False,
                'query': query,
                'rerank_enabled': False,
                'smart_rag_enabled': enable_smart_rag
            }

    def get_stats(self) -> dict:
        """获取系统统计信息"""
        try:
            source_files = set()
            for meta in self.metadata:
                if 'source_file' in meta:
                    source_files.add(meta['source_file'])
                elif 'metadata' in meta and 'source_file' in meta['metadata']:
                    source_files.add(meta['metadata']['source_file'])
            
            return {
                'total_documents': len(source_files),
                'total_chunks': len(self.metadata),
                'index_size': self.faiss_index.ntotal if self.faiss_index else 0,
                'chunker_type': self.chunker_type,
                'embedding_model': self.embedding_config.get('model_name', 'unknown'),
                'has_index': self.faiss_index is not None
            }
        except Exception as e:
            self.logger.error(f"获取统计信息失败: {e}")
            return {'error': str(e)}

    def clear_knowledge_base(self) -> dict:
        """清空知识库"""
        try:
            old_stats = self.get_stats()
            self.faiss_index = None
            self.metadata = []
            self.logger.info("知识库已清空")
            return {
                'success': True,
                'message': '知识库已清空',
                'cleared_documents': old_stats.get('total_documents', 0),
                'cleared_chunks': old_stats.get('total_chunks', 0)
            }
        except Exception as e:
            self.logger.error(f"清空知识库失败: {e}")
            return {
                'success': False,
                'message': str(e)
            }


    async def _check_query_relevance(self, query: str, threshold: float = 0.3) -> bool:
        """检查查询与知识库的相关性（改进版）"""
        try:
            # 添加关键词过滤
            common_greetings = ['你好', 'hello', 'hi', '您好', '早上好', '下午好', '晚上好']
            general_questions = ['什么是', '如何', '为什么', '怎么样', '怎么办']
            
            # 如果是常见问候语，直接返回不相关
            if any(greeting in query.lower() for greeting in common_greetings):
                self.logger.info(f"检测到问候语，跳过RAG检索: {query}")
                return False
                
            # 检查索引状态
            if self.faiss_index is None or len(self.metadata) == 0:
                self.logger.warning("知识库为空，跳过相关性检查")
                return False
            
            # 生成查询嵌入
            query_embedding = await self._generate_embeddings(query)
            if query_embedding is None:
                self.logger.error("生成查询嵌入失败")
                return True
            
            # 搜索最相似的文档
            search_k = min(3, len(self.metadata))
            scores, indices = self.faiss_index.search(query_embedding.reshape(1, -1), search_k)
            
            if len(scores[0]) == 0:
                return True
            
            # 提高判断标准
            max_score = float(scores[0][0])
            avg_score = float(np.mean(scores[0]))
            
            # 更严格的相关性判断
            is_relevant = (
                max_score >= threshold and  # 最高相似度必须达标
                avg_score >= threshold * 0.6  # 平均相似度也要较高
            )
            
            self.logger.info(
                f"相关性检查 - 查询: '{query[:30]}...', "
                f"最高相似度: {max_score:.4f}, 平均相似度: {avg_score:.4f}, "
                f"阈值: {threshold:.4f}, 结果: {'相关' if is_relevant else '不相关'}"
            )
            
            return is_relevant
            
        except Exception as e:
            self.logger.error(f"相关性检查失败: {e}")
            return True

    async def _extract_domain_keywords(self) -> List[str]:
        """从知识库中提取领域关键词（可选的增强方法）"""
        try:
            # 简单实现：从文档内容中提取高频词汇
            all_content = ' '.join([meta.get('content', '') for meta in self.metadata])
            # 这里可以使用更复杂的NLP技术提取关键词
            # 暂时返回空列表，后续可以扩展
            return []
        except Exception as e:
            self.logger.error(f"提取领域关键词失败: {e}")
            return []