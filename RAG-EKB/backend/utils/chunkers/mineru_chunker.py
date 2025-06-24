from typing import List, Dict, Any, Optional
from .base_chunker import BaseChunker, TextChunk, ChunkType
from ..mineru_processor import MinerUPDFProcessor, MinerUConfig
from utils.logs_utils import LoggerConfig

class MinerUChunker(BaseChunker):
    """基于MinerU的PDF分块器"""
    
    def __init__(self, chunk_size: int = 1000, overlap: int = 100, 
                 parse_method: str = 'auto', enable_formula: bool = True,
                 enable_table: bool = True, **kwargs):
        super().__init__()
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.logger = LoggerConfig().get_logger()
        
        # MinerU配置
        self.mineru_config = MinerUConfig(
            parse_method=parse_method,
            enable_formula=enable_formula,
            enable_table=enable_table,
            **kwargs
        )
        
        self.processor = MinerUPDFProcessor(self.mineru_config)
    
    def chunk(self, text: str, source_file: str = None, **kwargs) -> List[TextChunk]:
        """使用MinerU进行PDF分块"""
        chunks = []
        
        try:
            if source_file and source_file.lower().endswith('.pdf'):
                # 使用MinerU处理PDF
                result = self.processor.process_pdf_with_mineru(
                    source_file, 
                    kwargs.get('output_dir')
                )
                
                # 将元素转换为分块
                chunks = self._elements_to_chunks(result['elements'], source_file)
            else:
                # 普通文本分块
                chunks = self._chunk_text_simple(text, source_file)
            
            self.logger.info(f"MinerU分块完成，生成 {len(chunks)} 个分块")
            return chunks
            
        except Exception as e:
            self.logger.error(f"MinerU分块失败: {e}")
            # 降级到简单文本分块
            return self._chunk_text_simple(text, source_file)
    
    def _elements_to_chunks(self, elements: List, source_file: str) -> List[TextChunk]:
        """将PDF元素转换为文本分块"""
        chunks = []
        current_chunk_content = []
        current_chunk_size = 0
        
        for element in elements:
            element_content = self._format_element_content(element)
            element_size = len(element_content)
            
            # 检查是否需要开始新分块
            if (current_chunk_size + element_size > self.chunk_size and 
                current_chunk_content):
                
                # 创建当前分块
                chunk_text = "\n".join(current_chunk_content)
                chunks.append(TextChunk(
                    content=chunk_text,
                    chunk_type=ChunkType.MINERU,
                    source_file=source_file,
                    metadata={
                        'page_number': getattr(element, 'page_number', 1),
                        'chunk_method': 'mineru',
                        'element_count': len(current_chunk_content)
                    }
                ))
                
                # 开始新分块（保留重叠）
                if self.overlap > 0 and current_chunk_content:
                    overlap_content = current_chunk_content[-1:]
                    current_chunk_content = overlap_content + [element_content]
                    current_chunk_size = sum(len(c) for c in current_chunk_content)
                else:
                    current_chunk_content = [element_content]
                    current_chunk_size = element_size
            else:
                current_chunk_content.append(element_content)
                current_chunk_size += element_size
        
        # 处理最后一个分块
        if current_chunk_content:
            chunk_text = "\n".join(current_chunk_content)
            chunks.append(TextChunk(
                content=chunk_text,
                chunk_type=ChunkType.MINERU,
                source_file=source_file,
                metadata={
                    'chunk_method': 'mineru',
                    'element_count': len(current_chunk_content)
                }
            ))
        
        return chunks
    
    def _format_element_content(self, element) -> str:
        """格式化元素内容"""
        if element.element_type == 'header':
            return f"# {element.content}"
        elif element.element_type == 'table':
            return f"\n[表格]\n{element.content}\n"
        elif element.element_type == 'image':
            return f"\n[图像] {element.content}\n"
        elif element.element_type == 'formula':
            return f"\n$$\n{element.content}\n$$\n"
        else:
            return element.content
    
    def _chunk_text_simple(self, text: str, source_file: str) -> List[TextChunk]:
        """简单文本分块（降级方案）"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]
            
            chunks.append(TextChunk(
                content=chunk_text,
                chunk_type=ChunkType.MINERU,
                source_file=source_file,
                metadata={'chunk_method': 'simple_fallback'}
            ))
            
            start = end - self.overlap if self.overlap > 0 else end
        
        return chunks