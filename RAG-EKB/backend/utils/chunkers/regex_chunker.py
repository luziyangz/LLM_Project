import re
from typing import List, Optional, Pattern
from .base_chunker import BaseChunker, TextChunk, ChunkType

class RegexChunker(BaseChunker):
    """基于正则表达式的文本分块器"""
    
    def __init__(self, pattern: str = None, chunk_size: int = 1000, overlap: int = 100, 
                 min_chunk_size: int = 50, merge_small_chunks: bool = True, **kwargs):
        super().__init__(chunk_size, overlap, **kwargs)
        
        # 默认正则模式：按段落、标题、列表等结构分割
        if pattern is None:
            # 匹配段落分隔符、标题、列表项等
            pattern = r'(?:\n\s*\n|(?=\n\s*[#]+\s)|(?=\n\s*\d+\.|\n\s*[-*+]\s)|(?=\n\s*[一二三四五六七八九十]+[、.])|(?=\n\s*\([一二三四五六七八九十]+\)))'
        
        self.pattern = pattern
        self.compiled_pattern: Pattern = re.compile(pattern, re.MULTILINE | re.UNICODE)
        self.min_chunk_size = min_chunk_size
        self.merge_small_chunks = merge_small_chunks
    
    def chunk(self, text: str, source_file: Optional[str] = None) -> List[TextChunk]:
        """使用正则表达式进行文本分块"""
        if not text or len(text.strip()) == 0:
            return []
        
        text = text.strip()
        
        # 使用正则表达式分割文本
        raw_chunks = self._split_by_regex(text)
        
        # 处理分块：合并小块、处理重叠等
        processed_chunks = self._process_chunks(raw_chunks, text)
        
        # 创建TextChunk对象
        result_chunks = []
        for i, (content, start_pos, end_pos) in enumerate(processed_chunks):
            if content.strip():
                metadata = self._create_chunk_metadata(
                    chunk_id=i,
                    start_pos=start_pos,
                    end_pos=end_pos,
                    content=content,
                    chunk_type=ChunkType.SEMANTIC,  # 正则分块通常是语义相关的
                    source_file=source_file
                )
                
                chunk = TextChunk(content=content.strip(), metadata=metadata)
                if self.validate_chunk(chunk):
                    result_chunks.append(chunk)
        
        return result_chunks
    
    def _split_by_regex(self, text: str) -> List[str]:
        """使用正则表达式分割文本"""
        # 分割文本，保留分隔符
        parts = self.compiled_pattern.split(text)
        
        # 过滤空字符串
        chunks = [part.strip() for part in parts if part.strip()]
        
        return chunks
    
    def _process_chunks(self, raw_chunks: List[str], original_text: str) -> List[tuple]:
        """处理原始分块：合并小块、添加重叠、计算位置"""
        if not raw_chunks:
            return []
        
        processed = []
        current_pos = 0
        
        i = 0
        while i < len(raw_chunks):
            chunk_content = raw_chunks[i]
            
            # 查找当前块在原文中的位置
            start_pos = original_text.find(chunk_content, current_pos)
            if start_pos == -1:
                # 如果找不到，跳过这个块
                i += 1
                continue
            
            end_pos = start_pos + len(chunk_content)
            
            # 如果块太小且启用合并，尝试与下一块合并
            if (self.merge_small_chunks and 
                len(chunk_content) < self.min_chunk_size and 
                i + 1 < len(raw_chunks)):
                
                # 合并下一个块
                next_chunk = raw_chunks[i + 1]
                next_start = original_text.find(next_chunk, end_pos)
                
                if next_start != -1:
                    # 包含中间的文本
                    merged_content = original_text[start_pos:next_start + len(next_chunk)]
                    end_pos = next_start + len(next_chunk)
                    chunk_content = merged_content
                    i += 1  # 跳过下一个块
            
            # 如果块仍然太大，按固定长度进一步分割
            if len(chunk_content) > self.chunk_size:
                sub_chunks = self._split_large_chunk(chunk_content, start_pos)
                processed.extend(sub_chunks)
            else:
                processed.append((chunk_content, start_pos, end_pos))
            
            current_pos = end_pos
            i += 1
        
        # 添加重叠处理
        if self.overlap > 0:
            processed = self._add_overlap(processed, original_text)
        
        return processed
    
    def _split_large_chunk(self, content: str, base_start_pos: int) -> List[tuple]:
        """将过大的块进一步分割"""
        sub_chunks = []
        start = 0
        
        while start < len(content):
            end = min(start + self.chunk_size, len(content))
            
            # 尝试在句子边界分割
            if end < len(content):
                sentence_endings = ['。', '！', '？', '.', '!', '?', '\n']
                for i in range(end - 1, max(start, end - 200), -1):
                    if content[i] in sentence_endings:
                        end = i + 1
                        break
            
            sub_content = content[start:end]
            if sub_content.strip():
                sub_chunks.append((
                    sub_content,
                    base_start_pos + start,
                    base_start_pos + end
                ))
            
            start = max(start + 1, end - self.overlap)
            if start >= len(content):
                break
        
        return sub_chunks
    
    def _add_overlap(self, chunks: List[tuple], original_text: str) -> List[tuple]:
        """为分块添加重叠内容"""
        if len(chunks) <= 1:
            return chunks
        
        overlapped_chunks = []
        
        for i, (content, start_pos, end_pos) in enumerate(chunks):
            new_start = start_pos
            new_end = end_pos
            
            # 向前扩展（与前一块重叠）
            if i > 0 and self.overlap > 0:
                new_start = max(0, start_pos - self.overlap // 2)
            
            # 向后扩展（与后一块重叠）
            if i < len(chunks) - 1 and self.overlap > 0:
                new_end = min(len(original_text), end_pos + self.overlap // 2)
            
            # 提取新的内容
            new_content = original_text[new_start:new_end]
            overlapped_chunks.append((new_content, new_start, new_end))
        
        return overlapped_chunks


class CustomRegexChunker(RegexChunker):
    """自定义正则表达式分块器"""
    
    def __init__(self, custom_patterns: List[str], **kwargs):
        # 将多个模式组合成一个
        combined_pattern = '|'.join(f'({pattern})' for pattern in custom_patterns)
        super().__init__(pattern=combined_pattern, **kwargs)
        self.custom_patterns = custom_patterns


# 预定义的常用正则分块器
class ParagraphRegexChunker(RegexChunker):
    """段落分块器"""
    
    def __init__(self, **kwargs):
        # 匹配段落分隔符（两个或更多换行符）
        pattern = r'\n\s*\n+'
        super().__init__(pattern=pattern, **kwargs)


class SentenceRegexChunker(RegexChunker):
    """句子分块器"""
    
    def __init__(self, **kwargs):
        # 匹配句子结束标点
        pattern = r'[。！？.!?]+\s*'
        super().__init__(pattern=pattern, **kwargs)


class HeadingRegexChunker(RegexChunker):
    """标题分块器"""
    
    def __init__(self, **kwargs):
        # 匹配Markdown标题和中文标题
        pattern = r'(?=\n\s*#{1,6}\s+|\n\s*[一二三四五六七八九十]+[、.]|\n\s*\d+\.|\n\s*第[一二三四五六七八九十]+[章节部分])'  
        super().__init__(pattern=pattern, **kwargs)