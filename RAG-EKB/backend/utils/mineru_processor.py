import os
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging
from pathlib import Path

try:
    from magic_pdf.pipe.UNIPipe import UNIPipe
    from magic_pdf.pipe.OCRPipe import OCRPipe 
    from magic_pdf.pipe.TXTPipe import TXTPipe
    from magic_pdf.rw.DiskReaderWriter import DiskReaderWriter
    MINERU_AVAILABLE = True
except ImportError:
    MINERU_AVAILABLE = False
    # 创建模拟类以避免导入错误
    class UNIPipe:
        def __init__(self, *args, **kwargs):
            pass
        def pipe_parse(self):
            return {}
    
    class OCRPipe:
        def __init__(self, *args, **kwargs):
            pass
        def pipe_parse(self):
            return {}
    
    class TXTPipe:
        def __init__(self, *args, **kwargs):
            pass
        def pipe_parse(self):
            return {}
    
    class DiskReaderWriter:
        def __init__(self, *args, **kwargs):
            pass

from .pdf_processor import PDFElement

@dataclass
class MinerUConfig:
    """MinerU配置类"""
    parse_method: str = 'auto'  # auto, ocr, txt
    enable_formula: bool = True
    enable_table: bool = True
    enable_ocr: bool = True
    output_format: str = 'markdown'  # markdown, json
    device_mode: str = 'cpu'  # cpu, cuda
    lang: str = 'ch'  # 语言设置

class MinerUPDFProcessor:
    """基于MinerU的PDF处理器"""
    
    def __init__(self, config: Optional[MinerUConfig] = None):
        self.config = config or MinerUConfig()
        self.logger = logging.getLogger(__name__)
        
        if not MINERU_AVAILABLE:
            self.logger.warning("MinerU未安装，将使用模拟模式。请运行: pip install -U mineru[full]")
    
    def process_pdf_with_mineru(self, filepath: str, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """使用MinerU处理PDF文档"""
        try:
            if not MINERU_AVAILABLE:
                # 降级到基础PDF处理
                return self._fallback_processing(filepath)
            
            # 准备输出路径
            pdf_name = Path(filepath).stem
            if output_dir:
                output_path = Path(output_dir) / pdf_name
            else:
                output_path = Path(filepath).parent / pdf_name
            
            output_path.mkdir(parents=True, exist_ok=True)
            
            # 设置输出路径
            output_image_path = output_path / "images"
            output_image_path.mkdir(exist_ok=True)
            
            # 初始化读写器
            image_writer = DiskReaderWriter(str(output_image_path))
            
            # 根据解析方法选择管道
            if self.config.parse_method == 'ocr':
                pipe = OCRPipe(
                    pdf_path=filepath,
                    output_dir=str(output_path),
                    image_writer=image_writer
                )
            elif self.config.parse_method == 'txt':
                pipe = TXTPipe(
                    pdf_path=filepath,
                    output_dir=str(output_path),
                    image_writer=image_writer
                )
            else:  # auto
                pipe = UNIPipe(
                    pdf_path=filepath,
                    output_dir=str(output_path),
                    image_writer=image_writer
                )
            
            # 执行解析
            pipe_result = pipe.pipe_parse()
            
            # 转换为标准格式
            elements = self._convert_mineru_result_to_elements(pipe_result, filepath)
            
            return {
                'elements': elements,
                'total_pages': len(pipe_result.get('pdf_info', {}).get('pages', [])),
                'processing_method': f'mineru_{self.config.parse_method}',
                'mineru_result': pipe_result,
                'output_path': str(output_path)
            }
            
        except Exception as e:
            self.logger.error(f"MinerU处理PDF失败: {e}")
            # 降级处理
            return self._fallback_processing(filepath)
    
    def _fallback_processing(self, filepath: str) -> Dict[str, Any]:
        """降级处理方案"""
        from .pdf_processor import EnhancedPDFProcessor
        
        self.logger.info("使用降级PDF处理方案")
        processor = EnhancedPDFProcessor()
        return processor.process_pdf_comprehensive(filepath)
    
    def _convert_mineru_result_to_elements(self, mineru_result: Dict, source_file: str) -> List[PDFElement]:
        """将MinerU结果转换为标准PDFElement格式"""
        elements = []
        
        try:
            # 处理页面内容
            pages = mineru_result.get('pdf_info', {}).get('pages', [])
            
            for page_num, page_data in enumerate(pages, 1):
                # 处理文本块
                text_blocks = page_data.get('text_blocks', [])
                for block in text_blocks:
                    elements.append(PDFElement(
                        element_type='text',
                        content=block.get('text', ''),
                        page_number=page_num,
                        bbox=tuple(block.get('bbox', [0, 0, 0, 0])),
                        metadata={
                            'confidence': block.get('confidence', 0),
                            'block_type': block.get('type', 'text'),
                            'font_info': block.get('font_info', {})
                        }
                    ))
                
                # 处理表格
                tables = page_data.get('tables', [])
                for i, table in enumerate(tables):
                    table_text = self._format_table_content(table)
                    elements.append(PDFElement(
                        element_type='table',
                        content=table_text,
                        page_number=page_num,
                        bbox=tuple(table.get('bbox', [0, 0, 0, 0])),
                        metadata={
                            'table_index': i,
                            'table_structure': table.get('structure', {}),
                            'confidence': table.get('confidence', 0)
                        }
                    ))
                
                # 处理图像
                images = page_data.get('images', [])
                for i, image in enumerate(images):
                    image_content = self._format_image_content(image)
                    elements.append(PDFElement(
                        element_type='image',
                        content=image_content,
                        page_number=page_num,
                        bbox=tuple(image.get('bbox', [0, 0, 0, 0])),
                        metadata={
                            'image_index': i,
                            'image_path': image.get('image_path', ''),
                            'ocr_text': image.get('ocr_text', ''),
                            'description': image.get('description', '')
                        }
                    ))
                
                # 处理数学公式
                formulas = page_data.get('formulas', [])
                for i, formula in enumerate(formulas):
                    elements.append(PDFElement(
                        element_type='formula',
                        content=formula.get('latex', ''),
                        page_number=page_num,
                        bbox=tuple(formula.get('bbox', [0, 0, 0, 0])),
                        metadata={
                            'formula_index': i,
                            'confidence': formula.get('confidence', 0),
                            'formula_type': formula.get('type', 'inline')
                        }
                    ))
        
        except Exception as e:
            self.logger.error(f"转换MinerU结果失败: {e}")
        
        return elements
    
    def _format_table_content(self, table_data: Dict) -> str:
        """格式化表格内容"""
        try:
            if 'cells' in table_data:
                cells = table_data['cells']
                return self._cells_to_markdown(cells)
            elif 'text' in table_data:
                return table_data['text']
            else:
                return "[表格内容]"
        except Exception:
            return "[表格解析失败]"
    
    def _format_image_content(self, image_data: Dict) -> str:
        """格式化图像内容"""
        content_parts = []
        
        if image_data.get('ocr_text'):
            content_parts.append(f"图像中的文字：{image_data['ocr_text']}")
        
        if image_data.get('description'):
            content_parts.append(f"图像描述：{image_data['description']}")
        
        if not content_parts:
            content_parts.append("[图像内容]")
        
        return "\n".join(content_parts)
    
    def _cells_to_markdown(self, cells: List[List[str]]) -> str:
        """将单元格数据转换为markdown表格"""
        if not cells:
            return ""
        
        markdown_lines = []
        
        if cells:
            header = " | ".join(str(cell) for cell in cells[0])
            markdown_lines.append(header)
            separator = " | ".join(["---"] * len(cells[0]))
            markdown_lines.append(separator)
        
        for row in cells[1:]:
            row_text = " | ".join(str(cell) for cell in row)
            markdown_lines.append(row_text)
        
        return "\n".join(markdown_lines)