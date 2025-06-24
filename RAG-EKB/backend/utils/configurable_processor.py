import os
import asyncio
from typing import Dict, Any, List
from .document_processor import DocumentProcessor
from .rag_processor import RAGProcessor
import logging

logger = logging.getLogger(__name__)

class ConfigurableDocumentProcessor:
    """
    可配置的文档处理器，支持用户自定义处理策略
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.base_processor = DocumentProcessor()
        self.rag_processor = RAGProcessor()
        
    async def process_document_async(self, file_path: str) -> Dict[str, Any]:
        """
        异步处理文档
        """
        try:
            logger.info(f"开始配置化处理文档: {file_path}")
            
            # 1. 文档解析
            logger.info("步骤1: 解析文档")
            doc_info = self.base_processor.process_document(file_path)
            
            # 2. 预处理
            if self.config.get('preprocessing', {}).get('clean_text', True):
                logger.info("步骤2: 清理文本")
                doc_info = await self._clean_text(doc_info)
            
            # 3. 文本分块
            logger.info("步骤3: 文本分块")
            chunks = await self._chunk_text(doc_info['content'])
            
            # 4. 去重处理
            if self.config.get('preprocessing', {}).get('remove_duplicates', True):
                logger.info("步骤4: 去除重复内容")
                chunks = await self._remove_duplicates(chunks)
            
            # 5. 向量化
            logger.info("步骤5: 生成向量")
            embeddings = await self._generate_embeddings(chunks)
            
            # 6. 构建索引
            logger.info("步骤6: 构建索引")
            index_info = await self._build_index(chunks, embeddings)
            
            result = {
                "document_path": file_path,
                "chunk_count": len(chunks),
                "vector_dimension": len(embeddings[0]) if embeddings else 0,
                "index_size": index_info.get('size', 0),
                "processing_time": index_info.get('processing_time', 0),
                "config_used": self.config
            }
            
            logger.info(f"文档处理完成: {result}")
            return result
            
        except Exception as e:
            logger.error(f"配置化文档处理失败: {str(e)}", exc_info=True)
            raise
    
    async def _clean_text(self, doc_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        清理文本
        """
        # 实现文本清理逻辑
        content = doc_info['content']
        
        # 移除多余的空白字符
        content = ' '.join(content.split())
        
        # 移除特殊字符（根据配置）
        if self.config.get('preprocessing', {}).get('remove_special_chars', False):
            import re
            content = re.sub(r'[^\w\s\u4e00-\u9fff]', '', content)
        
        doc_info['content'] = content
        return doc_info
    
    async def _chunk_text(self, content: str) -> List[str]:
        """
        根据配置进行文本分块
        """
        chunking_config = self.config.get('chunking', {})
        method = chunking_config.get('method', 'fixed_size')
        chunk_size = chunking_config.get('chunk_size', 500)
        overlap_size = chunking_config.get('overlap_size', 50)
        
        if method == 'fixed_size':
            return self._fixed_size_chunking(content, chunk_size, overlap_size)
        elif method == 'semantic':
            return await self._semantic_chunking(content, chunk_size)
        elif method == 'sentence':
            return self._sentence_chunking(content, chunk_size)
        elif method == 'paragraph':
            return self._paragraph_chunking(content, chunk_size)
        else:
            return self._fixed_size_chunking(content, chunk_size, overlap_size)
    
    def _fixed_size_chunking(self, content: str, chunk_size: int, overlap_size: int) -> List[str]:
        """
        固定大小分块
        """
        chunks = []
        start = 0
        
        while start < len(content):
            end = start + chunk_size
            chunk = content[start:end]
            chunks.append(chunk)
            start = end - overlap_size
            
            if start >= len(content):
                break
        
        return chunks
    
    async def _semantic_chunking(self, content: str, max_chunk_size: int) -> List[str]:
        """
        语义分块（简化实现）
        """
        # 这里可以集成更复杂的语义分块算法
        sentences = content.split('。')
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk + sentence) <= max_chunk_size:
                current_chunk += sentence + "。"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + "。"
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _sentence_chunking(self, content: str, max_chunk_size: int) -> List[str]:
        """
        句子分块
        """
        sentences = content.split('。')
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk + sentence) <= max_chunk_size:
                current_chunk += sentence + "。"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + "。"
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _paragraph_chunking(self, content: str, max_chunk_size: int) -> List[str]:
        """
        段落分块
        """
        paragraphs = content.split('\n\n')
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            if len(current_chunk + paragraph) <= max_chunk_size:
                current_chunk += paragraph + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = paragraph + "\n\n"
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    async def _remove_duplicates(self, chunks: List[str]) -> List[str]:
        """
        去除重复的文本块
        """
        seen = set()
        unique_chunks = []
        
        for chunk in chunks:
            chunk_hash = hash(chunk.strip())
            if chunk_hash not in seen:
                seen.add(chunk_hash)
                unique_chunks.append(chunk)
        
        logger.info(f"去重前: {len(chunks)} 块，去重后: {len(unique_chunks)} 块")
        return unique_chunks
    
    async def _generate_embeddings(self, chunks: List[str]) -> List[List[float]]:
        """
        生成向量嵌入
        """
        embedding_config = self.config.get('embedding', {})
        batch_size = embedding_config.get('batch_size', 10)
        
        # 这里应该调用实际的嵌入模型
        # 暂时返回模拟数据
        embeddings = []
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            # 模拟向量生成
            batch_embeddings = [[0.1] * 1536 for _ in batch]
            embeddings.extend(batch_embeddings)
            
            # 模拟处理延迟
            await asyncio.sleep(0.1)
        
        return embeddings
    
    async def _build_index(self, chunks: List[str], embeddings: List[List[float]]) -> Dict[str, Any]:
        """
        构建向量索引
        """
        indexing_config = self.config.get('indexing', {})
        index_type = indexing_config.get('type', 'faiss_flat')
        
        # 这里应该调用实际的索引构建逻辑
        # 暂时返回模拟数据
        await asyncio.sleep(1)  # 模拟索引构建时间
        
        return {
            "type": index_type,
            "size": len(embeddings) * 1536 * 4,  # 假设每个float 4字节
            "processing_time": 2.5
        }