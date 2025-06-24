from typing import List, Dict, Any, Optional
from .base_chunker import BaseChunker, TextChunk, ChunkMetadata, ChunkType
import re

class DocxChunker(BaseChunker):
    """DOCX文档专用分块器"""
    
    def __init__(self, chunk_size: int = 1200, overlap: int = 150, 
                 preserve_headers: bool = True, preserve_tables: bool = True,
                 min_chunk_size: int = 100, merge_small_chunks: bool = True):
        super().__init__()
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.preserve_headers = preserve_headers
        self.preserve_tables = preserve_tables
        self.min_chunk_size = min_chunk_size
        self.merge_small_chunks = merge_small_chunks
    
    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[TextChunk]:
        """对DOCX文档进行智能分块"""
        chunks = []
        
        # 解析文档结构
        sections = self._parse_docx_structure(text)
        
        chunk_id = 0
        for section in sections:
            section_chunks = self._chunk_section(section, chunk_id)
            chunks.extend(section_chunks)
            chunk_id += len(section_chunks)
        
        return chunks
    
    def _parse_docx_structure(self, text: str) -> List[Dict[str, Any]]:
        """解析DOCX文档结构"""
        sections = []
        lines = text.split('\n')
        
        current_section = {
            'type': 'content',
            'title': None,
            'content': [],
            'metadata': {}
        }
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # 检测标题
            if line.startswith('#'):
                # 保存当前section
                if current_section['content']:
                    sections.append(current_section)
                
                # 开始新section
                current_section = {
                    'type': 'header',
                    'title': line,
                    'content': [line],
                    'metadata': {'level': len(line) - len(line.lstrip('#'))}
                }
            
            # 检测表格
            elif line.startswith('[表格'):
                if current_section['content']:
                    sections.append(current_section)
                
                current_section = {
                    'type': 'table',
                    'title': None,
                    'content': [line],
                    'metadata': {'is_table': True}
                }
            
            # 检测页眉页脚
            elif line.startswith('[页眉]') or line.startswith('[页脚]'):
                current_section['metadata']['has_header_footer'] = True
                current_section['content'].append(line)
            
            else:
                current_section['content'].append(line)
        
        # 添加最后一个section
        if current_section['content']:
            sections.append(current_section)
        
        return sections
    
    def _chunk_section(self, section: Dict[str, Any], start_chunk_id: int) -> List[TextChunk]:
        """对单个section进行分块"""
        content = '\n'.join(section['content'])
        
        # 表格和标题保持完整
        if section['type'] in ['table', 'header'] or len(content) <= self.chunk_size:
            return [self._create_chunk(content, start_chunk_id, section)]
        
        # 长内容进行分块
        chunks = []
        sentences = self._split_sentences(content)
        
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            if current_length + sentence_length > self.chunk_size and current_chunk:
                # 创建当前块
                chunk_content = ' '.join(current_chunk)
                chunks.append(self._create_chunk(chunk_content, start_chunk_id + len(chunks), section))
                
                # 处理重叠
                overlap_sentences = self._get_overlap_sentences(current_chunk)
                current_chunk = overlap_sentences
                current_length = sum(len(s) for s in current_chunk)
            
            current_chunk.append(sentence)
            current_length += sentence_length
        
        # 添加最后一个块
        if current_chunk:
            chunk_content = ' '.join(current_chunk)
            chunks.append(self._create_chunk(chunk_content, start_chunk_id + len(chunks), section))
        
        return chunks
    
    def _create_chunk(self, content: str, chunk_id: int, section: Dict[str, Any]) -> TextChunk:
        """创建文本块"""
        metadata = ChunkMetadata(
            chunk_id=chunk_id,
            start_pos=0,  # 在实际使用中需要计算
            end_pos=len(content),
            char_count=len(content),
            word_count=len(content.split()),
            chunk_type=ChunkType.FIXED_LENGTH,  # 需要添加DOCX类型
            section_title=section.get('title'),
            confidence_score=1.0
        )
        
        # 添加section特定的元数据
        for key, value in section.get('metadata', {}).items():
            setattr(metadata, key, value)
        
        return TextChunk(content=content, metadata=metadata)
    
    def _split_sentences(self, text: str) -> List[str]:
        """分割句子"""
        # 中英文句子分割
        sentences = re.split(r'[。！？.!?]\s*', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _get_overlap_sentences(self, sentences: List[str]) -> List[str]:
        """获取重叠的句子"""
        overlap_chars = 0
        overlap_sentences = []
        
        for sentence in reversed(sentences):
            if overlap_chars + len(sentence) <= self.overlap:
                overlap_sentences.insert(0, sentence)
                overlap_chars += len(sentence)
            else:
                break
        
        return overlap_sentences