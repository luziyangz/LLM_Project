from typing import List, Optional
from langchain.text_splitter import CharacterTextSplitter
from .base_chunker import BaseChunker, TextChunk, ChunkType

class LangChainCharacterChunker(BaseChunker):
    """基于LangChain CharacterTextSplitter的分块器"""
    
    def __init__(self, chunk_size: int = 1000, overlap: int = 100, 
                 separator: str = "\n\n", **kwargs):
        super().__init__(chunk_size, overlap, **kwargs)
        self.separator = separator
        
        # 初始化LangChain的CharacterTextSplitter
        self.text_splitter = CharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separator=separator,
            length_function=len,
            is_separator_regex=False
        )
    
    def chunk(self, text: str, source_file: Optional[str] = None) -> List[TextChunk]:
        """使用LangChain CharacterTextSplitter进行分块"""
        try:
            # 使用LangChain进行分块
            chunks = self.text_splitter.split_text(text)
            
            result_chunks = []
            current_pos = 0
            
            for i, chunk_content in enumerate(chunks):
                # 查找当前分块在原文中的位置
                start_pos = text.find(chunk_content, current_pos)
                if start_pos == -1:
                    start_pos = current_pos
                
                end_pos = start_pos + len(chunk_content)
                
                # 创建分块元数据
                metadata = self._create_chunk_metadata(
                    chunk_id=i,
                    start_pos=start_pos,
                    end_pos=end_pos,
                    content=chunk_content,
                    chunk_type=ChunkType.LANGCHAIN_CHARACTER,
                    source_file=source_file
                )
                
                # 创建文本分块
                text_chunk = TextChunk(
                    content=chunk_content,
                    metadata=metadata
                )
                
                # 验证分块有效性
                if self.validate_chunk(text_chunk):
                    result_chunks.append(text_chunk)
                
                current_pos = end_pos
            
            return result_chunks
            
        except Exception as e:
            raise RuntimeError(f"LangChain分块失败: {str(e)}")
    
    def set_separator(self, separator: str):
        """动态设置分隔符"""
        self.separator = separator
        self.text_splitter = CharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.overlap,
            separator=separator,
            length_function=len,
            is_separator_regex=False
        )