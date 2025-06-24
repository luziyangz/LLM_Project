from typing import List, Optional
from .base_chunker import BaseChunker, TextChunk, ChunkType

class FixedLengthChunker(BaseChunker):
    """固定长度分块器"""
    
    def __init__(self, chunk_size: int = 1000, overlap: int = 100, 
                 preserve_sentences: bool = True, min_chunk_size: int = 50):
        super().__init__(chunk_size, overlap)
        self.preserve_sentences = preserve_sentences
        self.min_chunk_size = min_chunk_size
    
    def chunk(self, text: str, source_file: Optional[str] = None) -> List[TextChunk]:
        """固定长度分块"""
        if not text or len(text.strip()) == 0:
            return []
        
        text = text.strip()
        chunks = []
        start_pos = 0
        chunk_id = 0
        
        while start_pos < len(text):
            end_pos = min(start_pos + self.chunk_size, len(text))
            
            # 如果启用句子保护且不是最后一块，尝试在句子边界分割
            if self.preserve_sentences and end_pos < len(text):
                end_pos = self._find_sentence_boundary(text, start_pos, end_pos)
            
            chunk_content = text[start_pos:end_pos].strip()
            
            # 跳过太小的分块
            if len(chunk_content) < self.min_chunk_size and start_pos > 0:
                break
            
            if chunk_content:
                metadata = self._create_chunk_metadata(
                    chunk_id=chunk_id,
                    start_pos=start_pos,
                    end_pos=end_pos,
                    content=chunk_content,
                    chunk_type=ChunkType.FIXED_LENGTH,
                    source_file=source_file
                )
                
                chunk = TextChunk(content=chunk_content, metadata=metadata)
                if self.validate_chunk(chunk):
                    chunks.append(chunk)
                    chunk_id += 1
            
            # 计算下一个分块的起始位置（考虑重叠）
            if end_pos >= len(text):
                break
            start_pos = max(start_pos + 1, end_pos - self.overlap)
        
        return chunks
    
    def _find_sentence_boundary(self, text: str, start_pos: int, end_pos: int) -> int:
        """寻找句子边界"""
        # 中英文句子结束标点
        sentence_endings = ['。', '！', '？', '.', '!', '?', '\n\n']
        
        # 从end_pos向前搜索句子结束标点
        search_start = max(start_pos, end_pos - 200)  # 限制搜索范围
        
        for i in range(end_pos - 1, search_start - 1, -1):
            if text[i] in sentence_endings:
                # 确保不会产生太小的分块
                if i - start_pos >= self.min_chunk_size:
                    return i + 1
        
        return end_pos