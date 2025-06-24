from typing import List, Dict, Any
from .base_chunker import BaseChunker, ChunkType
from ..pdf_processor import EnhancedPDFProcessor, PDFElement
from utils.logs_utils import LoggerConfig

class MultimodalChunker(BaseChunker):
    """多模态分块器，支持图文混合内容的智能分块"""
    
    def __init__(self, chunk_size: int = 1000, overlap: int = 100, 
                 context_window: int = 200, openai_config: Dict = None):
        super().__init__()
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.context_window = context_window  # 图像前后文本的上下文窗口
        self.logger = LoggerConfig().get_logger()
        
        # 初始化PDF处理器
        self.pdf_processor = EnhancedPDFProcessor(openai_config)
    
    def chunk(self, text: str, source_file: str = None, **kwargs) -> List:
        """对多模态内容进行分块"""
        chunks = []
        
        try:
            # 如果是PDF文件，进行结构化处理
            if source_file and source_file.lower().endswith('.pdf'):
                chunks = self._chunk_pdf_multimodal(source_file, **kwargs)
            else:
                # 普通文本分块
                chunks = self._chunk_text_simple(text, source_file)
            
            self.logger.info(f"多模态分块完成，生成 {len(chunks)} 个分块")
            return chunks
            
        except Exception as e:
            self.logger.error(f"多模态分块失败: {e}")
            return self._chunk_text_simple(text, source_file)
    
    def _chunk_pdf_multimodal(self, filepath: str, **kwargs) -> List:
        """对PDF进行多模态分块"""
        chunks = []
        
        # 使用增强PDF处理器提取所有元素
        pdf_result = self.pdf_processor.process_pdf_comprehensive(filepath)
        elements = pdf_result.get('elements', [])
        
        if not elements:
            return chunks
        
        # 按页面分组元素
        pages = self._group_elements_by_page(elements)
        
        for page_num, page_elements in pages.items():
            page_chunks = self._process_page_elements(page_elements, page_num, filepath)
            chunks.extend(page_chunks)
        
        return chunks
    
    def _group_elements_by_page(self, elements: List[PDFElement]) -> Dict[int, List[PDFElement]]:
        """按页面分组元素"""
        pages = {}
        for element in elements:
            page_num = element.page_number
            if page_num not in pages:
                pages[page_num] = []
            pages[page_num].append(element)
        
        # 按位置排序每页的元素
        for page_num in pages:
            pages[page_num].sort(key=lambda x: (x.bbox[1], x.bbox[0]))  # 按y坐标，然后x坐标排序
        
        return pages
    
    def _process_page_elements(self, elements: List[PDFElement], 
                             page_num: int, source_file: str) -> List:
        """处理单页的元素，生成智能分块"""
        chunks = []
        current_chunk_content = []
        current_chunk_size = 0
        
        for i, element in enumerate(elements):
            element_content = self._format_element_content(element)
            element_size = len(element_content)
            
            # 检查是否需要开始新分块
            if (current_chunk_size + element_size > self.chunk_size and 
                current_chunk_content):
                
                # 创建当前分块
                chunk = self._create_chunk_from_content(
                    current_chunk_content, page_num, source_file
                )
                chunks.append(chunk)
                
                # 处理重叠
                current_chunk_content = self._handle_overlap(current_chunk_content)
                current_chunk_size = sum(len(content) for content, _ in current_chunk_content)
            
            # 添加当前元素
            current_chunk_content.append((element_content, element))
            current_chunk_size += element_size
        
        # 处理最后一个分块
        if current_chunk_content:
            chunk = self._create_chunk_from_content(
                current_chunk_content, page_num, source_file
            )
            chunks.append(chunk)
        
        return chunks
    
    def _format_element_content(self, element: PDFElement) -> str:
        """格式化元素内容"""
        if element.element_type == 'image':
            return f"[图像内容] {element.content}"
        elif element.element_type == 'table':
            return f"[表格内容]\n{element.content}"
        elif element.element_type == 'header':
            return f"# {element.content}"
        else:
            return element.content
    
    def _create_chunk_from_content(self, content_list: List, 
                                 page_num: int, source_file: str):
        """从内容列表创建分块"""
        # 组合内容
        combined_content = "\n\n".join([content for content, _ in content_list])
        
        # 收集元数据
        element_types = [element.element_type for _, element in content_list]
        has_image = 'image' in element_types
        has_table = 'table' in element_types
        has_text = any(t in ['text', 'header'] for t in element_types)
        
        # 创建分块元数据
        metadata = self._create_chunk_metadata(
            chunk_id=f"multimodal_{page_num}_{len(content_list)}",
            source_file=source_file,
            start_pos=0,
            end_pos=len(combined_content),
            chunk_type=ChunkType.MULTIMODAL,
            char_count=len(combined_content),
            word_count=len(combined_content.split()),
            additional_metadata={
                'page_number': page_num,
                'element_count': len(content_list),
                'has_image': has_image,
                'has_table': has_table,
                'has_text': has_text,
                'element_types': element_types
            }
        )
        
        # 创建分块对象
        from .base_chunker import Chunk
        return Chunk(content=combined_content, metadata=metadata)
    
    def _handle_overlap(self, content_list: List) -> List:
        """处理分块重叠"""
        if not content_list or self.overlap <= 0:
            return []
        
        # 保留最后几个元素作为重叠
        overlap_content = []
        current_size = 0
        
        for content, element in reversed(content_list):
            if current_size + len(content) <= self.overlap:
                overlap_content.insert(0, (content, element))
                current_size += len(content)
            else:
                break
        
        return overlap_content
    
    def _chunk_text_simple(self, text: str, source_file: str = None) -> List:
        """简单文本分块（回退方案）"""
        chunks = []
        words = text.split()
        
        for i in range(0, len(words), self.chunk_size - self.overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk_content = ' '.join(chunk_words)
            
            metadata = self._create_chunk_metadata(
                chunk_id=f"text_{i}",
                source_file=source_file,
                start_pos=i,
                end_pos=i + len(chunk_words),
                chunk_type=ChunkType.FIXED_LENGTH,
                char_count=len(chunk_content),
                word_count=len(chunk_words)
            )
            
            from .base_chunker import Chunk
            chunks.append(Chunk(content=chunk_content, metadata=metadata))
        
        return chunks