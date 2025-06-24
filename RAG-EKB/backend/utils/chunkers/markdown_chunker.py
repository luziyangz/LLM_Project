from typing import List, Optional, Dict, Any
import re
from .base_chunker import BaseChunker, TextChunk, ChunkType

class MarkdownChunker(BaseChunker):
    """Markdown文档分块器，支持基于Markdown结构的智能分块"""
    
    def __init__(self, chunk_size: int = 1000, overlap: int = 100, 
                 respect_headers: bool = True, include_metadata: bool = True, **kwargs):
        super().__init__(chunk_size, overlap, **kwargs)
        self.respect_headers = respect_headers  # 是否尊重标题结构
        self.include_metadata = include_metadata  # 是否包含元数据信息
        
        # Markdown元素的正则表达式
        self.patterns = {
            'header': r'^(#{1,6})\s+(.+)$',  # 标题
            'code_block': r'```[\s\S]*?```',  # 代码块
            'inline_code': r'`[^`]+`',  # 行内代码
            'list_item': r'^\s*[-*+]\s+(.+)$',  # 列表项
            'numbered_list': r'^\s*\d+\.\s+(.+)$',  # 有序列表
            'blockquote': r'^>\s+(.+)$',  # 引用
            'horizontal_rule': r'^\s*[-*_]{3,}\s*$',  # 分隔线
            'table_row': r'^\|.*\|$',  # 表格行
            'link': r'\[([^\]]+)\]\(([^)]+)\)',  # 链接
            'image': r'!\[([^\]]*)\]\(([^)]+)\)'  # 图片
        }
    
    def chunk(self, text: str, source_file: Optional[str] = None) -> List[TextChunk]:
        """对Markdown文本进行分块"""
        if not text or len(text.strip()) == 0:
            return []
        
        text = text.strip()
        
        # 解析Markdown结构
        sections = self._parse_markdown_structure(text)
        
        # 根据结构进行分块
        chunks = self._create_chunks_from_sections(sections, text)
        
        # 创建TextChunk对象
        result_chunks = []
        for i, (content, start_pos, end_pos, section_info) in enumerate(chunks):
            if content.strip():
                metadata = self._create_chunk_metadata(
                    chunk_id=i,
                    start_pos=start_pos,
                    end_pos=end_pos,
                    content=content,
                    chunk_type=ChunkType.MARKDOWN,
                    source_file=source_file
                )
                
                # 添加Markdown特定的元数据
                if section_info:
                    metadata.section_title = section_info.get('title')
                    metadata.confidence_score = section_info.get('confidence', 1.0)
                
                result_chunks.append(TextChunk(content=content, metadata=metadata))
        
        return result_chunks
    
    def _parse_markdown_structure(self, text: str) -> List[Dict[str, Any]]:
        """解析Markdown文档结构"""
        lines = text.split('\n')
        sections = []
        current_section = {
            'type': 'content',
            'level': 0,
            'title': None,
            'start_line': 0,
            'content_lines': []
        }
        
        for i, line in enumerate(lines):
            # 检查是否是标题
            header_match = re.match(self.patterns['header'], line)
            if header_match and self.respect_headers:
                # 保存当前section
                if current_section['content_lines']:
                    current_section['end_line'] = i - 1
                    sections.append(current_section.copy())
                
                # 开始新的section
                level = len(header_match.group(1))
                title = header_match.group(2).strip()
                current_section = {
                    'type': 'header',
                    'level': level,
                    'title': title,
                    'start_line': i,
                    'content_lines': [line]
                }
            else:
                current_section['content_lines'].append(line)
        
        # 添加最后一个section
        if current_section['content_lines']:
            current_section['end_line'] = len(lines) - 1
            sections.append(current_section)
        
        return sections
    
    def _create_chunks_from_sections(self, sections: List[Dict[str, Any]], 
                                   original_text: str) -> List[tuple]:
        """根据解析的sections创建分块"""
        chunks = []
        current_chunk_content = ""
        current_chunk_start = 0
        current_section_info = None
        
        for section in sections:
            section_content = '\n'.join(section['content_lines'])
            
            # 如果当前section是标题，且当前chunk不为空，先保存当前chunk
            if (section['type'] == 'header' and current_chunk_content.strip() and 
                len(current_chunk_content) > self.chunk_size // 2):
                
                end_pos = current_chunk_start + len(current_chunk_content)
                chunks.append((
                    current_chunk_content.strip(),
                    current_chunk_start,
                    end_pos,
                    current_section_info
                ))
                
                # 开始新的chunk
                current_chunk_start = original_text.find(section_content, end_pos)
                if current_chunk_start == -1:
                    current_chunk_start = end_pos
                current_chunk_content = section_content
                current_section_info = {
                    'title': section.get('title'),
                    'level': section.get('level'),
                    'confidence': 1.0
                }
            else:
                # 添加到当前chunk
                if current_chunk_content:
                    current_chunk_content += '\n' + section_content
                else:
                    current_chunk_content = section_content
                    current_chunk_start = original_text.find(section_content)
                    if current_chunk_start == -1:
                        current_chunk_start = 0
                
                # 更新section信息
                if section['type'] == 'header':
                    current_section_info = {
                        'title': section.get('title'),
                        'level': section.get('level'),
                        'confidence': 1.0
                    }
            
            # 检查是否需要分割过大的chunk
            if len(current_chunk_content) > self.chunk_size:
                split_chunks = self._split_large_chunk(
                    current_chunk_content, current_chunk_start, current_section_info
                )
                chunks.extend(split_chunks[:-1])  # 添加除最后一个外的所有chunk
                
                # 更新当前chunk为最后一个分割的chunk
                if split_chunks:
                    last_chunk = split_chunks[-1]
                    current_chunk_content = last_chunk[0]
                    current_chunk_start = last_chunk[1]
                    current_section_info = last_chunk[3]
        
        # 添加最后一个chunk
        if current_chunk_content.strip():
            end_pos = current_chunk_start + len(current_chunk_content)
            chunks.append((
                current_chunk_content.strip(),
                current_chunk_start,
                end_pos,
                current_section_info
            ))
        
        return self._add_overlap(chunks)
    
    def _split_large_chunk(self, content: str, start_pos: int, 
                          section_info: Dict[str, Any]) -> List[tuple]:
        """分割过大的chunk，尽量保持Markdown结构完整"""
        chunks = []
        lines = content.split('\n')
        current_chunk_lines = []
        current_length = 0
        
        for line in lines:
            line_length = len(line) + 1  # +1 for newline
            
            # 检查是否是重要的Markdown元素（不应该被分割）
            is_code_block = line.strip().startswith('```')
            is_header = re.match(self.patterns['header'], line)
            is_list_item = (re.match(self.patterns['list_item'], line) or 
                          re.match(self.patterns['numbered_list'], line))
            
            # 如果添加这一行会超过chunk_size，且当前chunk不为空
            if (current_length + line_length > self.chunk_size and 
                current_chunk_lines and not is_code_block):
                
                # 保存当前chunk
                chunk_content = '\n'.join(current_chunk_lines)
                chunk_start = start_pos
                chunk_end = chunk_start + len(chunk_content)
                
                chunks.append((chunk_content, chunk_start, chunk_end, section_info))
                
                # 开始新的chunk
                start_pos = chunk_end + 1
                current_chunk_lines = [line]
                current_length = line_length
            else:
                current_chunk_lines.append(line)
                current_length += line_length
        
        # 添加最后一个chunk
        if current_chunk_lines:
            chunk_content = '\n'.join(current_chunk_lines)
            chunk_end = start_pos + len(chunk_content)
            chunks.append((chunk_content, start_pos, chunk_end, section_info))
        
        return chunks
    
    def _add_overlap(self, chunks: List[tuple]) -> List[tuple]:
        """为chunks添加重叠内容"""
        if not chunks or self.overlap <= 0:
            return chunks
        
        overlapped_chunks = []
        
        for i, (content, start_pos, end_pos, section_info) in enumerate(chunks):
            final_content = content
            final_start = start_pos
            
            # 添加前向重叠
            if i > 0 and self.overlap > 0:
                prev_content = chunks[i-1][0]
                overlap_text = prev_content[-self.overlap:]
                
                # 尝试在单词边界处截断
                space_pos = overlap_text.find(' ')
                if space_pos > 0:
                    overlap_text = overlap_text[space_pos:]
                
                final_content = overlap_text + '\n' + content
                final_start = start_pos - len(overlap_text) - 1
            
            overlapped_chunks.append((final_content, final_start, end_pos, section_info))
        
        return overlapped_chunks
    
    def get_markdown_statistics(self, text: str) -> Dict[str, Any]:
        """获取Markdown文档的统计信息"""
        stats = {
            'total_chars': len(text),
            'total_lines': len(text.split('\n')),
            'headers': [],
            'code_blocks': 0,
            'lists': 0,
            'links': 0,
            'images': 0
        }
        
        lines = text.split('\n')
        in_code_block = False
        
        for line in lines:
            # 检查代码块
            if line.strip().startswith('```'):
                if not in_code_block:
                    stats['code_blocks'] += 1
                in_code_block = not in_code_block
                continue
            
            if in_code_block:
                continue
            
            # 检查标题
            header_match = re.match(self.patterns['header'], line)
            if header_match:
                level = len(header_match.group(1))
                title = header_match.group(2).strip()
                stats['headers'].append({'level': level, 'title': title})
            
            # 检查列表
            if (re.match(self.patterns['list_item'], line) or 
                re.match(self.patterns['numbered_list'], line)):
                stats['lists'] += 1
            
            # 检查链接和图片
            stats['links'] += len(re.findall(self.patterns['link'], line))
            stats['images'] += len(re.findall(self.patterns['image'], line))
        
        return stats