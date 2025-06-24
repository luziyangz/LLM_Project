from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

class ChunkType(Enum):
    """分块类型枚举"""
    FIXED_LENGTH = "fixed_length"
    REGEX = "regex"
    CUSTOM_REGEX = "custom_regex"
    PARAGRAPH_REGEX = "paragraph_regex"
    SENTENCE_REGEX = "sentence_regex"
    HEADING_REGEX = "heading_regex"
    LANGCHAIN_CHARACTER = "langchain_character"
    PDF_STRUCTURE = "pdf_structure"
    MINERU = "mineru"
    MARKDOWN = "markdown"  # 添加Markdown类型
    # MULTIMODAL = "multimodal"  # 保持注释状态
    SLIDING_WINDOW = "sliding_window"
    # 删除重复的定义
    # LANGCHAIN_CHARACTER = "langchain_character"
    DOCX_STRUCTURE = "docx_structure"  # 添加DOCX类型

@dataclass
class ChunkMetadata:
    """分块元数据"""
    chunk_id: int
    start_pos: int
    end_pos: int
    char_count: int
    word_count: int
    chunk_type: ChunkType
    source_file: Optional[str] = None
    section_title: Optional[str] = None
    confidence_score: Optional[float] = None

@dataclass
class TextChunk:
    """文本分块数据结构"""
    content: str
    metadata: ChunkMetadata
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'content': self.content,
            'metadata': {
                'chunk_id': self.metadata.chunk_id,
                'start_pos': self.metadata.start_pos,
                'end_pos': self.metadata.end_pos,
                'char_count': self.metadata.char_count,
                'word_count': self.metadata.word_count,
                'chunk_type': self.metadata.chunk_type.value,
                'source_file': self.metadata.source_file,
                'section_title': self.metadata.section_title,
                'confidence_score': self.metadata.confidence_score
            }
        }

class BaseChunker(ABC):
    """分块器基类"""
    
    def __init__(self, chunk_size: int = 1000, overlap: int = 100, **kwargs):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.config = kwargs
    
    @abstractmethod
    def chunk(self, text: str, source_file: Optional[str] = None) -> List[TextChunk]:
        """分块方法，子类必须实现"""
        pass
    
    def _create_chunk_metadata(self, chunk_id: int, start_pos: int, end_pos: int, 
                              content: str, chunk_type: ChunkType, 
                              source_file: Optional[str] = None) -> ChunkMetadata:
        """创建分块元数据"""
        return ChunkMetadata(
            chunk_id=chunk_id,
            start_pos=start_pos,
            end_pos=end_pos,
            char_count=len(content),
            word_count=len(content.split()),
            chunk_type=chunk_type,
            source_file=source_file
        )
    
    def validate_chunk(self, chunk: TextChunk) -> bool:
        """验证分块是否有效"""
        return (
            len(chunk.content.strip()) > 0 and
            chunk.metadata.char_count > 0 and
            chunk.metadata.end_pos > chunk.metadata.start_pos
        )