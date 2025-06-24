import os
from typing import Dict, Type, List, Optional, Any
from .base_chunker import BaseChunker, TextChunk, ChunkType
from .fixed_length_chunker import FixedLengthChunker
from .regex_chunker import RegexChunker, CustomRegexChunker, ParagraphRegexChunker, SentenceRegexChunker, HeadingRegexChunker
from .langchain_chunker import LangChainCharacterChunker
from .markdown_chunker import MarkdownChunker  # 添加导入
# from .multimodal_chunker import MultimodalChunker  # 保持注释
# from .pdf_chunker import PDFStructureChunker  # 保持注释
from .docx_chunker import DocxChunker  # 添加导入

# 条件导入MinerU相关模块
try:
    from ..mineru_processor import MinerUPDFProcessor, MinerUConfig
    from .mineru_chunker import MinerUChunker
    MINERU_AVAILABLE = True
except ImportError:
    MINERU_AVAILABLE = False
    MinerUChunker = None
    MinerUPDFProcessor = None
    MinerUConfig = None

class ChunkManager:
    """分块管理器 - 统一管理各种分块策略"""
    
    def __init__(self):
        self._chunkers: Dict[str, Type[BaseChunker]] = {
            'fixed_length': FixedLengthChunker,
            'regex': RegexChunker,
            'custom_regex': CustomRegexChunker,
            'paragraph_regex': ParagraphRegexChunker,
            'sentence_regex': SentenceRegexChunker,
            'heading_regex': HeadingRegexChunker,
            'langchain_character': LangChainCharacterChunker,
            'markdown': MarkdownChunker,  # 添加Markdown分块器
            # 'multimodal': MultimodalChunker,  # 保持注释
            # 'pdf_structure': PDFStructureChunker  # 保持注释
        }
        
        # 条件添加MinerU分块器
        if MINERU_AVAILABLE:
            self._chunkers['mineru'] = MinerUChunker
        
        self._default_configs = {
            'fixed_length': {
                'chunk_size': 1000,
                'overlap': 100,
                'preserve_sentences': True,
                'min_chunk_size': 50
            },
            'regex': {
                'chunk_size': 1000,
                'overlap': 100,
                'min_chunk_size': 50,
                'merge_small_chunks': True
            },
            'paragraph_regex': {
                'chunk_size': 2000,
                'overlap': 50,
                'min_chunk_size': 100,
                'merge_small_chunks': True
            },
            'sentence_regex': {
                'chunk_size': 500,
                'overlap': 50,
                'min_chunk_size': 20,
                'merge_small_chunks': True
            },
            'heading_regex': {
                'chunk_size': 1500,
                'overlap': 100,
                'min_chunk_size': 100,
                'merge_small_chunks': True
            },
            'langchain_character': {
                'chunk_size': 1000,
                'overlap': 100,
                'separator': '\n\n'
            },
            'pdf_structure': {
                'chunk_size': 1000,
                'overlap': 100,
                'preserve_structure': True,
                'min_chunk_size': 50
            },
            'mineru': {
                'chunk_size': 1000,
                'overlap': 100,
                'parse_method': 'auto',
                'enable_formula': True,
                'enable_table': True,
                'enable_ocr': True,
                'device_mode': 'cpu',
                'lang': 'ch'
            },
            
            # 添加DOCX专用配置
            'docx_structure': FixedLengthChunker,  # 暂时使用固定长度，后续可创建专用分块器
        }
        
        self._default_configs = {
            'fixed_length': {
                'chunk_size': 1000,
                'overlap': 100,
                'preserve_sentences': True,
                'min_chunk_size': 50
            },
            'regex': {
                'chunk_size': 1000,
                'overlap': 100,
                'min_chunk_size': 50,
                'merge_small_chunks': True
            },
            'paragraph_regex': {
                'chunk_size': 2000,
                'overlap': 50,
                'min_chunk_size': 100,
                'merge_small_chunks': True
            },
            'sentence_regex': {
                'chunk_size': 500,
                'overlap': 50,
                'min_chunk_size': 20,
                'merge_small_chunks': True
            },
            'heading_regex': {
                'chunk_size': 1500,
                'overlap': 100,
                'min_chunk_size': 100,
                'merge_small_chunks': True
            },
            'langchain_character': {
                'chunk_size': 1000,
                'overlap': 100,
                'separator': '\n\n'
            },
            'pdf_structure': {
                'chunk_size': 1000,
                'overlap': 100,
                'preserve_structure': True,
                'min_chunk_size': 50
            },
            'mineru': {
                'chunk_size': 1000,
                'overlap': 100,
                'parse_method': 'auto',
                'enable_formula': True,
                'enable_table': True,
                'enable_ocr': True,
                'device_mode': 'cpu',
                'lang': 'ch'
            },
            
            # 添加DOCX专用配置
            'docx_structure': {
                'chunk_size': 1200,
                'overlap': 150,
                'preserve_headers': True,
                'preserve_tables': True,
                'include_metadata': True,
                'min_chunk_size': 100,
                'merge_small_chunks': True
            }
        }
        
        # 更新文件类型分块器映射
        self._file_type_chunkers = {
            '.docx': 'docx_structure',  # 使用DOCX专用分块器
            '.doc': 'docx_structure',
            '.pdf': 'pdf_structure',
            '.md': 'markdown',
            '.txt': 'fixed_length'
        }
    
    def get_optimal_chunker_for_file(self, file_path: str) -> str:
        """根据文件类型推荐最佳分块器"""
        ext = os.path.splitext(file_path)[1].lower()
        return self._file_type_chunkers.get(ext, 'fixed_length')
        
        # 添加Markdown分块器的默认配置
        if 'markdown' not in self._default_configs:
            self._default_configs['markdown'] = {
                'chunk_size': 1000,
                'overlap': 100,
                'respect_headers': True,
                'preserve_code_blocks': True,
                'include_metadata': True
            }
    
    def chunk_text(self, text: str, chunker_type: str = 'fixed_length', 
                   source_file: Optional[str] = None, **kwargs) -> List[TextChunk]:
        """对文本进行分块处理"""
        try:
            # 获取分块器类
            if chunker_type not in self._chunkers:
                raise ValueError(f"不支持的分块器类型: {chunker_type}")
            
            chunker_class = self._chunkers[chunker_type]
            
            # 获取默认配置并合并用户配置
            default_config = self._default_configs.get(chunker_type, {})
            config = {**default_config, **kwargs}
            
            # 创建分块器实例
            chunker = chunker_class(**config)
            
            # 执行分块
            chunks = chunker.chunk(text, source_file)
            
            return chunks
            
        except Exception as e:
            raise RuntimeError(f"分块处理失败: {str(e)}")
    
    def get_available_chunkers(self) -> List[str]:
        """获取可用的分块器列表"""
        return list(self._chunkers.keys())
    
    def get_chunker_config(self, chunker_type: str) -> Dict[str, Any]:
        """获取分块器的默认配置"""
        return self._default_configs.get(chunker_type, {})