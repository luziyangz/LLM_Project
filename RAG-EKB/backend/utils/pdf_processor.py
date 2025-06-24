import fitz  # PyMuPDF
# 注释掉图像处理相关导入
# from PIL import Image
# import io
# import base64
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import pdfplumber
import logging

@dataclass
class PDFElement:
    """PDF元素数据类"""
    element_type: str  # 'text', 'image', 'table', 'header', 'footer'
    content: str
    page_number: int
    bbox: tuple  # 边界框坐标
    metadata: Dict[str, Any]

# 注释掉图像处理器相关导入
# from .enhanced_image_processor import EnhancedImageProcessor, ImageContent

class EnhancedPDFProcessor:
    """增强的PDF处理器，支持文本、表格、布局分析（暂时移除图像处理功能）"""
    
    def __init__(self, openai_config: Optional[Dict] = None):
        self.supported_formats = ['.pdf']
        self.logger = logging.getLogger(__name__)
        # 注释掉图像处理器初始化
        # self.image_processor = EnhancedImageProcessor(openai_config)
    
    def process_pdf_comprehensive(self, filepath: str) -> Dict[str, Any]:
        """综合处理PDF文档（暂时不包含图像处理）"""
        elements = []
        
        # 使用pdfplumber进行结构化提取
        with pdfplumber.open(filepath) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                # 提取文本块
                text_elements = self._extract_text_blocks(page, page_num)
                elements.extend(text_elements)
                
                # 提取表格
                table_elements = self._extract_tables(page, page_num)
                elements.extend(table_elements)
        
        # 注释掉图像提取功能
        # image_elements = self._extract_images_pymupdf(filepath)
        # elements.extend(image_elements)
        
        return {
            'elements': elements,
            'total_pages': len(pdf.pages),
            'processing_method': 'comprehensive_no_images'
        }
    
    def _extract_text_blocks(self, page, page_num: int) -> List[PDFElement]:
        """提取文本块，区分标题、正文等"""
        elements = []
        
        # 获取文本对象
        chars = page.chars
        if not chars:
            return elements
        
        # 按字体大小和位置分组文本
        text_groups = self._group_text_by_style(chars)
        
        for group in text_groups:
            element_type = self._classify_text_type(group)
            content = ''.join([char['text'] for char in group])
            
            if content.strip():
                bbox = self._get_text_bbox(group)
                elements.append(PDFElement(
                    element_type=element_type,
                    content=content.strip(),
                    page_number=page_num,
                    bbox=bbox,
                    metadata={
                        'font_size': group[0].get('size', 0),
                        'font_name': group[0].get('fontname', ''),
                        'char_count': len(content.strip())
                    }
                ))
        
        return elements
    
    def _extract_tables(self, page, page_num: int) -> List[PDFElement]:
        """提取表格"""
        elements = []
        tables = page.extract_tables()
        
        for i, table in enumerate(tables):
            if table:
                # 转换表格为文本格式
                table_text = self._table_to_text(table)
                bbox = page.bbox  # 简化处理，实际应该获取表格具体位置
                
                elements.append(PDFElement(
                    element_type='table',
                    content=table_text,
                    page_number=page_num,
                    bbox=bbox,
                    metadata={
                        'table_index': i,
                        'rows': len(table),
                        'cols': len(table[0]) if table else 0
                    }
                ))
        
        return elements
    
    # 注释掉整个图像提取方法
    # def _extract_images_pymupdf(self, filepath: str) -> List[PDFElement]:
    #     """使用PyMuPDF提取图片并进行内容分析"""
    #     elements = []
    #     
    #     try:
    #         doc = fitz.open(filepath)
    #         
    #         for page_num in range(len(doc)):
    #             page = doc.load_page(page_num)
    #             image_list = page.get_images()
    #             
    #             for img_index, img in enumerate(image_list):
    #                 try:
    #                     # 提取图像数据
    #                     xref = img[0]
    #                     pix = fitz.Pixmap(doc, xref)
    #                     
    #                     if pix.n - pix.alpha < 4:  # 确保是RGB或灰度图像
    #                         # 转换为PNG格式的字节数据
    #                         img_data = pix.tobytes("png")
    #                         
    #                         # 使用增强图像处理器分析图像内容
    #                         image_content = self.image_processor.process_image_comprehensive(
    #                             img_data, 'PNG'
    #                         )
    #                         
    #                         # 组合OCR文字和AI描述
    #                         combined_content = self._combine_image_content(image_content)
    #                         
    #                         # 获取图像在页面中的位置
    #                         img_rect = page.get_image_rects(xref)
    #                         bbox = img_rect[0] if img_rect else page.rect
    #                         
    #                         elements.append(PDFElement(
    #                             element_type='image',
    #                             content=combined_content,
    #                             page_number=page_num + 1,
    #                             bbox=tuple(bbox),
    #                             metadata={
    #                                 'image_index': img_index,
    #                                 'image_format': 'PNG',
    #                                 'ocr_text': image_content.ocr_text,
    #                                 'ai_description': image_content.description,
    #                                 'confidence': image_content.confidence,
    #                                 'language': image_content.language,
    #                                 'size': image_content.size
    #                             }
    #                         ))
    #                     
    #                     pix = None  # 释放内存
    #                     
    #                 except Exception as e:
    #                     self.logger.warning(f"处理图像 {img_index} 失败: {e}")
    #                     continue
    #         
    #         doc.close()
    #         
    #     except Exception as e:
    #         self.logger.error(f"PyMuPDF图像提取失败: {e}")
    #     
    #     return elements
    
    # 注释掉图像内容组合方法
    # def _combine_image_content(self, image_content: ImageContent) -> str:
    #     """组合图像的OCR文字和AI描述"""
    #     content_parts = []
    #     
    #     if image_content.ocr_text.strip():
    #         content_parts.append(f"图像中的文字内容：{image_content.ocr_text}")
    #     
    #     if image_content.description and "失败" not in image_content.description:
    #         content_parts.append(f"图像描述：{image_content.description}")
    #     
    #     if not content_parts:
    #         content_parts.append("[图像内容]")
    #     
    #     return "\n".join(content_parts)
    
    def _group_text_by_style(self, chars: List[Dict]) -> List[List[Dict]]:
        """按字体样式分组文本"""
        if not chars:
            return []
        
        groups = []
        current_group = [chars[0]]
        
        for char in chars[1:]:
            # 检查是否应该开始新组（基于字体大小、样式等）
            if (abs(char.get('size', 0) - current_group[-1].get('size', 0)) > 1 or
                char.get('fontname', '') != current_group[-1].get('fontname', '')):
                groups.append(current_group)
                current_group = [char]
            else:
                current_group.append(char)
        
        if current_group:
            groups.append(current_group)
        
        return groups
    
    def _classify_text_type(self, text_group: List[Dict]) -> str:
        """分类文本类型（标题、正文等）"""
        if not text_group:
            return 'text'
        
        avg_size = sum(char.get('size', 0) for char in text_group) / len(text_group)
        text_content = ''.join([char['text'] for char in text_group]).strip()
        
        # 简单的分类逻辑
        if avg_size > 14:  # 大字体可能是标题
            return 'header'
        elif len(text_content) < 50 and text_content.isupper():
            return 'header'
        elif avg_size < 10:  # 小字体可能是页脚
            return 'footer'
        else:
            return 'text'
    
    def _get_text_bbox(self, text_group: List[Dict]) -> tuple:
        """获取文本组的边界框"""
        if not text_group:
            return (0, 0, 0, 0)
        
        x0 = min(char.get('x0', 0) for char in text_group)
        y0 = min(char.get('y0', 0) for char in text_group)
        x1 = max(char.get('x1', 0) for char in text_group)
        y1 = max(char.get('y1', 0) for char in text_group)
        
        return (x0, y0, x1, y1)
    
    def _table_to_text(self, table: List[List]) -> str:
        """将表格转换为文本格式"""
        if not table:
            return ""
        
        text_lines = []
        for row in table:
            # 过滤None值并转换为字符串
            clean_row = [str(cell) if cell is not None else "" for cell in row]
            text_lines.append(" | ".join(clean_row))
        
        return "\n".join(text_lines)