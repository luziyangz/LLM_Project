import os
import logging
from random import normalvariate
from typing import Dict, List, Any, Optional, Tuple, Iterable  # 添加Iterable导入
from pathlib import Path
from dataclasses import dataclass
import json
import time  # 添加time模块导入
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.backend.docling_parse_v2_backend import DoclingParseV2DocumentBackend
# 导入缺失的模块
from tabulate import tabulate  # 用于表格转Markdown

# 添加日志记录器
_log = logging.getLogger(__name__)


class PDFParser:
    def __init__(
        self,
        pdf_backend=DoclingParseV2DocumentBackend,  # PDF后端处理引擎
        output_dir: Path = Path("./parsed_pdfs"),  # 输出目录，默认为当前目录下的parsed_pdfs
        num_threads: int = None,  # 线程数，默认为None使用系统默认
        csv_metadata_path: Path = None,  # CSV元数据文件路径
    ):
        """初始化PDF解析器
        
        参数:
            pdf_backend: PDF解析后端引擎
            output_dir: 解析结果输出目录
            num_threads: 处理线程数
            csv_metadata_path: 包含元数据的CSV文件路径
        """
        self.pdf_backend = pdf_backend
        self.output_dir = output_dir
        self.doc_converter = self._create_document_converter()  # 创建文档转换器
        self.num_threads = num_threads
        self.metadata_lookup = {}  # 元数据查找字典
        self.debug_data_path = None  # 调试数据路径

        # 元数据控制 - 如果提供了CSV元数据路径，则解析元数据
        if csv_metadata_path is not None:
            self.metadata_lookup = self._parse_csv_metadata(csv_metadata_path)

        # 多线程控制 - 设置OpenMP线程数环境变量
        if self.num_threads is not None:
            os.environ["OMP_NUM_THREADS"] = str(self.num_threads)
            
    @staticmethod
    def _parse_csv_metadata(csv_path: Path) -> dict:
        """解析CSV元数据文件并创建查找字典
        
        该方法用于读取包含文件元数据的CSV文件，提取文件哈希值与公司名称的映射关系。
        支持新旧两种CSV格式，确保向后兼容性。
        
        Args:
            csv_path (Path): CSV元数据文件的路径
            
        Returns:
            dict: 以文件sha1哈希值为键，包含公司名称信息的字典
        """
        import csv  # 导入CSV处理模块
        metadata_lookup = {}  # 初始化元数据查找字典
        
        # 以UTF-8编码打开CSV文件，确保中文字符正确读取
        with open(csv_path, 'r', encoding='utf-8') as csvfile:
            # 使用DictReader将CSV行转换为字典，第一行作为字段名
            reader = csv.DictReader(csvfile)
            
            # 逐行处理CSV数据
            for row in reader:
                # 处理新旧两种CSV格式的公司名字段
                # 新格式使用'company_name'，旧格式使用'name'
                # 使用get()方法避免KeyError，提供默认值''
                company_name = row.get('company_name', row.get('name', '')).strip('"')
                
                # 以文件的sha1哈希值为键，存储公司名称信息
                # 构建嵌套字典结构，便于后续扩展其他元数据字段
                metadata_lookup[row['sha1']] = {
                    'company_name': company_name  # 去除双引号后的公司名称
                }
                
        return metadata_lookup  # 返回完整的元数据查找字典


    # 实现_create_document_converter 方法
    def _create_document_converter(self) -> DocumentConverter:
        """创建并配置文档转换器
        
        返回:
            DocumentConverter: 配置好的文档转换器实例，用于处理PDF文件
        """
        from docling.document_converter import DocumentConverter, FormatOption
        from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode, EasyOcrOptions
        from docling.datamodel.base_models import InputFormat
        from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline

        # 配置pipeline_options - 设置PDF处理管道选项
        pipeline_options = PdfPipelineOptions()
        # 启用OCR - 光学字符识别
        pipeline_options.do_ocr = True
        # 配置OCR识别中文和英文
        ocr_options = EasyOcrOptions(lang=['en'], force_full_page_ocr=False)        
        #ocr_options = EasyOcrOptions(lang=['ch_sim','en'], force_full_page_ocr=False)
        # 设置OCR选项
        pipeline_options.ocr_options = ocr_options
        # 启用表格结构识别
        pipeline_options.do_table_structure = True
        # 启用单元格匹配
        pipeline_options.table_structure_options.do_cell_matching = True
        # 设置表格识别模式为高精度模式
        pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE        

        # 配置格式选项 - 为PDF格式指定处理管道和后端
        format_options = {
            InputFormat.PDF: FormatOption(
                pipeline_cls=StandardPdfPipeline,  # 使用标准PDF处理管道
                pipeline_options=pipeline_options,  # 应用上面配置的选项
                backend=self.pdf_backend  # 使用指定的PDF后端
            )
        }

        # 返回配置好的文档转换器
        return DocumentConverter(format_options=format_options)
    
    """
    文档转换与处理流程：
    - convert_documents : 调用文档转换器批量处理PDF文件
    - process_documents : 处理转换结果，生成结构化JSON输出
    - _normalize_page_sequence : 确保页码连续性的内部工具方法
    - parse_and_export : 单线程处理流程的入口方法
    - parse_and_export_parallel : 多进程并行处理的高性能版本
    """

    # 实现process_documents，处理PDF文件
    def process_documents(self, cov_results: Iterable):
        """处理转换结果，生成结构化JSON输出
        
        参数:
            cov_results: 文档转换结果的可迭代对象
            
        返回:
            tuple: (失败数量, 成功数量)
        """
        # 创建输出目录
        if self.output_dir is not None: 
            self.output_dir.mkdir(parents=True, exist_ok=True)  

        failed_count = 0  # 失败计数器
        success_count = 0  # 成功计数器
        
        # 遍历转换结果
        for conv_result in cov_results:
            try:
                if conv_result.status == ConversionStatus.SUCCESS:  # 如果转换成功
                    success_count += 1
                    # 创建JSON报告处理器实例
                    # 意义：封装报告生成逻辑，保持代码模块化
                    proccessor = JsonReportProcessor(metadata_lookup=self.metadata_lookup, debug_data_path=self.debug_data_path)
                    
                    # 将数据转换成字典
                    data = conv_result.document.export_to_dic()
                    # 确保页码的连续性
                    normalized_data = self._normalize_page_sequence(data)
                    # 组装处理后的报告
                    proccessed_report = proccessor.assemble_report(conv_result, normalized_data)
                    doc_filename = conv_result.input.file.stem
                    # 如果输出目录存在，将处理后的报告保存为JSON文件
                    if self.output_dir is not None:
                        output_path = self.output_dir / f"{doc_filename}.json"
                        with open(output_path, "w", encoding="utf-8") as f:
                            json.dump(proccessed_report, f, ensure_ascii=False, indent=2)

                else:  # 如果转换失败
                    # 处理转换失败的情况
                    failed_count += 1
                    logging.error(f"文档处理失败: {conv_result.error}")
                
            except Exception as e:  # 捕获处理过程中的异常
                failed_count += 1
                logging.error(f"处理文档时发生错误: {str(e)}")
                
        return failed_count, success_count  # 返回处理结果统计



    # 实现convert_documents 文档转换
    def convert_documents(self, input_doc_paths: List[Path]) -> Iterable:
        """批量转换PDF文档
        
        参数:
            input_doc_paths: PDF文件路径列表
            
        返回:
            Iterable: 转换结果的可迭代对象
        """
        cov_results = self.doc_converter.convert_all(source=input_doc_paths)
        return cov_results

    # 实现单线程处理流程方法
    def parse_and_export(self, input_doc_paths: List[Path] = None, doc_dir: Path = None):
        """单线程处理PDF文件并导出结果
        
        参数:
            input_doc_paths: PDF文件路径列表，与doc_dir二选一
            doc_dir: 包含PDF文件的目录，与input_doc_paths二选一
            
        返回:
            dict: 处理结果统计信息
        """
        # 添加时间戳，记录开始时间
        start_time = time.time()
        
        # 参数验证 - 确保至少提供了一种输入方式
        if input_doc_paths is None and doc_dir is None:
            raise ValueError("必须提供input_doc_paths或doc_dir参数之一。")
            
        # 如果提供了目录而非文件列表，则获取目录中的所有PDF文件
        if input_doc_paths is None and doc_dir is not None:
            input_doc_paths = list(doc_dir.glob("*.pdf"))

        total_docs = len(input_doc_paths)  # 总文档数
        # 添加日志，记录开始处理
        logging.info(f"开始处理 {total_docs} 个文档")

        # 调用方法convert_documents 转换文档，并返回转换结果
        conv_results = self.convert_documents(input_doc_paths)
        # 添加日志，记录转换完成
        logging.info(f"文档转换完成，共处理 {total_docs} 个文档")

        # 调用方法process_documents 处理转换结果，生成结构化JSON文件
        failed_count, success_count = self.process_documents(conv_results)
        # 添加日志，记录处理完成
        logging.info(f"文档处理完成，共处理 {total_docs} 个文档，成功 {success_count} 个，失败 {failed_count} 个")
        
        # 计算总耗时
        elapsed_time = time.time() - start_time

        # 如果有失败的文档，记录详细信息
        if failed_count > 0:
            failed_docs = [doc for doc, result in conv_results.items() if not result.success]
            failed_info = "\n".join([f"- {doc.name}: {conv_results[doc].error}" for doc in failed_docs])
            logging.error(f"以下 {failed_count} 个文档处理失败:\n{failed_info}")
            
        # 返回处理结果统计
        return {
            "total": total_docs,  # 总文档数
            "success": success_count,  # 成功处理数
            "failed": failed_count,  # 失败数
            "failed_docs": [  # 失败文档详情
                {"path": str(doc), "error": conv_results[doc].error} 
                for doc in conv_results if not conv_results[doc].success
            ] if failed_count > 0 else [],
            "elapsed_time": elapsed_time  # 总耗时(秒)
        }

    def parse_and_export_parallel(
        self,
        input_doc_paths: List[Path] = None,
        doc_dir: Path = None,
        optimal_workers: int = 10,
        chunk_size: int = None
    ):
        """并行处理PDF文件
        
        使用多进程并行处理大量PDF文件，提高处理效率。
        
        参数:
            input_doc_paths: PDF文件路径列表
            doc_dir: 包含PDF文件的目录
            optimal_workers: 工作进程数量，默认为10
            chunk_size: 每个进程处理的文档数量
        """
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor, as_completed

        # 获取输入路径（如果未提供）
        if input_doc_paths is None and doc_dir is not None:
            input_doc_paths = list(doc_dir.glob("*.pdf"))

        total_pdfs = len(input_doc_paths)  # 总PDF文件数
        _log.info(f"开始并行处理 {total_pdfs} 个文档")
        
        # 获取CPU核心数
        cpu_count = multiprocessing.cpu_count()
        
        # 计算最佳工作进程数（如果未指定）
        if optimal_workers is None:
            optimal_workers = min(cpu_count, total_pdfs)  # 不超过CPU核心数和文档数
        
        # 计算每个进程处理的文档数（如果未指定）
        if chunk_size is None:
            # 确保至少为1
            chunk_size = max(1, total_pdfs // optimal_workers)
        
        # 将文档分成多个块
        chunks = [
            input_doc_paths[i : i + chunk_size]
            for i in range(0, total_pdfs, chunk_size)
        ]

        # 记录开始时间
        start_time = time.time()
        processed_count = 0  # 已处理文档计数
        
        # 使用ProcessPoolExecutor进行并行处理
        with ProcessPoolExecutor(max_workers=optimal_workers) as executor:
            # 调度所有任务
            futures = [
                executor.submit(
                    _process_chunk,  # 处理文档块的函数
                    chunk,  # 文档块
                    self.pdf_backend,  # PDF后端
                    self.output_dir,  # 输出目录
                    self.num_threads,  # 线程数
                    self.metadata_lookup,  # 元数据
                    self.debug_data_path  # 调试数据路径
                )
                for chunk in chunks
            ]
            
            # 等待完成并记录结果
            for future in as_completed(futures):
                try:
                    result = future.result()
                    # 从结果中提取处理的PDF数量
                    processed_count += int(result.split()[1])  
                    _log.info(f"{'#'*50}\n{result} ({processed_count}/{total_pdfs} 总计)\n{'#'*50}")
                except Exception as e:
                    _log.error(f"处理文档块时出错: {str(e)}")
                    raise  # 重新抛出异常

        # 计算总耗时
        elapsed_time = time.time() - start_time
        _log.info(f"并行处理完成，耗时 {elapsed_time:.2f} 秒。")


    def _normalize_page_sequence(self, data: dict) -> dict:
        '''确保页码的连续性
        
        处理文档数据，确保页码从1开始连续，对缺失的页码创建空页面。
        
        参数:
            data: 文档数据字典
            
        返回:
            dict: 规范化后的文档数据
        '''
        # 如果数据中没有content字段，直接返回原数据
        if 'content' not in data:
            return data
            
        # 复制原始数据，避免修改原始对象
        normalized_data = data.copy()

        # 获取现有页码并确定最大值
        existing_pages = {page['page'] for page in data['content']}
        max_page = max(existing_pages)

        # 创建空页面模板 - 用于填充缺失的页面
        empty_page_template = {
            "content": [],  # 空内容列表
            "page_dimensions": {}  # 空页面尺寸
        }

        # 构建规范化内容（关键逻辑）
        # 确保页码从1到max_page连续存在
        new_content = []
        for page_num in range(1, max_page + 1):
            # 查找现有页面或创建新的空页面
            page_content = next(
                (page for page in data['content'] if page['page'] == page_num),
                {"page": page_num, **empty_page_template}  # 如果页面不存在，创建空页面
            )
            new_content.append(page_content) 

        # 更新规范化数据的content字段
        normalized_data['content'] = new_content
        return normalized_data




class JsonReportProcessor:
    """JSON报告处理器
    
    负责将PDF解析结果组装成结构化的JSON报告格式。
    处理元数据、内容、表格和图片等不同类型的数据。
    """
    def __init__(self, metadata_lookup: dict = None, debug_data_path = None):
        """初始化JSON报告处理器
        
        参数:
            metadata_lookup: 元数据查找字典
            debug_data_path: 调试数据保存路径
        """
        self.metadata_lookup = metadata_lookup  # 元数据查找字典
        self.debug_data_path = debug_data_path  # 调试数据路径
    

    # 组装完整报告的主方法
    def assemble_report(self, conv_result, normalized_data=None):
        """组装完整的JSON报告
        
        参数:
            conv_result: 转换结果对象
            normalized_data: 规范化的数据，如果为None则从conv_result中提取
            
        返回:
            dict: 组装好的完整报告
        """
        # 优先使用传入的normalized_data，否则从转换结果中导出数据
        data = normalized_data if normalized_data is not None else conv_result.document.export_to_dic()
        assembled_report = {}  # 初始化报告字典

        # 组装各部分数据
        assembled_report['metainfo'] = self.assemble_metainfo(data)  # 元数据
        assembled_report['content'] = self.assemble_content(data)  # 内容
        assembled_report['tables'] = self.assemble_tables(conv_result.document.tables, data)  # 表格
        assembled_report['pictures'] = self.assemble_pictures(data)  # 图片
        
        return assembled_report  # 返回组装好的报告

    def assemble_metainfo(self, data):
        """组装元数据信息
        
        从数据中提取元数据，并与CSV元数据结合。
        
        参数:
            data: 文档数据
            
        返回:
            dict: 元数据字典
        """
        metainfo = {}  # 初始化元数据字典
        
        # 提取SHA1哈希值
        if 'sha1' in data['origin']:
            metainfo['sha1'] = data['origin']['sha1']
            
        # 如果存在元数据查找表且SHA1在其中，添加公司名称
        if self.metadata_lookup and metainfo.get('sha1') in self.metadata_lookup:
            csv_meta = self.metadata_lookup[metainfo['sha1']]
            metainfo['company_name'] = csv_meta['company_name']
            
        return metainfo

    def process_table(self, table_data):
        """处理表格数据
        
        参数:
            table_data: 表格数据
            
        返回:
            str: 处理后的表格内容
        """
        # 实现表格处理逻辑
        return 'processed_table_content'            


    def debug_data(self, data):
        """保存调试数据
        
        将数据保存为JSON文件用于调试。
        
        参数:
            data: 要保存的数据
        """
        # 如果未设置调试数据路径，直接返回
        if self.debug_data_path is None:
            return
            
        # 获取文档名称
        doc_name = data['name']
        # 构建保存路径
        path = self.debug_data_path / f"{doc_name}.json"
        # 创建目录（如果不存在）
        path.parent.mkdir(parents=True, exist_ok=True)    
        # 保存数据为JSON文件
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def assemble_content(self, data):
        """组装文档内容
        
        处理文档主体内容，包括文本、表格和图片引用。
        按页面组织内容，确保页面顺序正确。
        
        参数:
            data: 文档数据
            
        返回:
            list: 按页面组织的内容列表
        """
        pages = {}  # 页面字典，键为页码
        
        # 展开body children以包含组引用
        body_children = data['body']['children']
        groups = data.get('groups', [])
        expanded_body_children = self.expand_groups(body_children, groups)

        # 处理主体内容
        for item in expanded_body_children:
            # 处理引用类型的项目
            if isinstance(item, dict) and '$ref' in item:
                ref = item['$ref']  # 获取引用路径
                ref_type, ref_num = ref.split('/')[-2:]  # 解析引用类型和编号
                ref_num = int(ref_num)  # 转换为整数

                # 处理文本引用
                if ref_type == 'texts':
                    text_item = data['texts'][ref_num]  # 获取文本项
                    content_item = self._process_text_reference(ref_num, data)  # 处理文本引用

                    # 如果项目有组ID，添加组信息
                    if 'group_id' in item:
                        content_item['group_id'] = item['group_id']
                        content_item['group_name'] = item['group_name']
                        content_item['group_label'] = item['group_label']

                    # 从prov获取页码
                    if 'prov' in text_item and text_item['prov']:
                        page_num = text_item['prov'][0]['page_no']

                        # 如果页面不存在，初始化页面
                        if page_num not in pages:
                            pages[page_num] = {
                                'page': page_num,
                                'content': [],
                                'page_dimensions': text_item['prov'][0].get('bbox', {})
                            }

                        # 将内容项添加到页面
                        pages[page_num]['content'].append(content_item)
                
                # 处理表格引用
                elif ref_type == 'tables':
                    table_item = data['tables'][ref_num]  # 获取表格项
                    content_item = {
                        'type': 'table',
                        'table_id': ref_num
                    }

                    # 从prov获取页码
                    if 'prov' in table_item and table_item['prov']:
                        page_num = table_item['prov'][0]['page_no']

                        # 如果页面不存在，初始化页面
                        if page_num not in pages:
                            pages[page_num] = {
                                'page': page_num,
                                'content': [],
                                'page_dimensions': table_item['prov'][0].get('bbox', {})
                            }

                        # 将表格项添加到页面
                        pages[page_num]['content'].append(content_item)

                # 处理图片引用
                elif ref_type == 'pictures':
                    picture_item = data['pictures'][ref_num]  # 获取图片项
                    content_item = {
                        'type': 'picture',
                        'picture_id': ref_num
                    }
                    
                    # 从prov获取页码
                    if 'prov' in picture_item and picture_item['prov']:
                        page_num = picture_item['prov'][0]['page_no']

                        # 如果页面不存在，初始化页面
                        if page_num not in pages:
                            pages[page_num] = {
                                'page': page_num,
                                'content': [],
                                'page_dimensions': picture_item['prov'][0].get('bbox', {})
                            }
                        
                        # 将图片项添加到页面
                        pages[page_num]['content'].append(content_item)  # 注意：原代码中有拼写错误，应为content_item

        # 按页码排序页面
        sorted_pages = [pages[page_num] for page_num in sorted(pages.keys())]
        return sorted_pages

    def _process_text_reference(self, ref_num, data):
        """处理文本引用并创建内容项
        
        参数:
            ref_num: 文本引用编号
            data: 文档数据字典
            
        返回:
            dict: 处理后的内容项，包含文本信息
        """
        text_item = data['texts'][ref_num]  # 获取文本项
        item_type = text_item['label']  # 获取文本类型标签
        
        # 创建内容项字典
        content_item = {
            'text': text_item.get('text', ''),  # 文本内容
            'type': item_type,  # 类型
            'text_id': ref_num  # 文本ID
        }
        
        # 仅当原始内容与处理后内容不同时，添加'orig'字段
        orig_content = text_item.get('orig', '')
        if orig_content != text_item.get('text', ''):
            content_item['orig'] = orig_content

        # 添加额外字段（如果存在）
        if 'enumerated' in text_item:  # 枚举标记
            content_item['enumerated'] = text_item['enumerated']
        if 'marker' in text_item:  # 标记
            content_item['marker'] = text_item['marker']
            
        return content_item
    


    def assemble_tables(self, tables, data):
        """组装表格数据
        
        处理文档中的所有表格，生成结构化表格数据。
        
        参数:
            tables: 表格对象列表
            data: 文档数据
            
        返回:
            list: 组装好的表格列表
        """
        assembled_tables = []  # 初始化表格列表
        
        # 处理每个表格
        for i, table in enumerate(tables):
            # 转换表格为JSON对象
            table_json_obj = table.model_dump()
            # 转换表格为Markdown格式
            table_md = self._table_to_md(table_json_obj)
            # 转换表格为HTML格式
            table_html = table.export_to_html()
            
            # 获取表格数据
            table_data = data['tables'][i]
            # 获取表格页码
            table_page_num = table_data['prov'][0]['page_no']
            # 获取表格边界框
            table_bbox = table_data['prov'][0]['bbox']
            # 转换边界框为列表格式
            table_bbox = [
                table_bbox['l'],  # 左
                table_bbox['t'],  # 上
                table_bbox['r'],  # 右
                table_bbox['b']   # 下
            ]
            
            # 获取表格行列数
            nrows = table_data['data']['num_rows']  # 行数
            ncols = table_data['data']['num_cols']  # 列数

            # 获取表格引用编号
            ref_num = table_data['self_ref'].split('/')[-1]
            ref_num = int(ref_num)

            # 创建表格对象
            table_obj = {
                'table_id': ref_num,  # 表格ID
                'page': table_page_num,  # 页码
                'bbox': table_bbox,  # 边界框
                '#-rows': nrows,  # 行数
                '#-cols': ncols,  # 列数
                'markdown': table_md,  # Markdown格式
                'html': table_html,  # HTML格式
                'json': table_json_obj  # JSON格式
            }
            # 添加到表格列表
            assembled_tables.append(table_obj)
            
        return assembled_tables

    def _table_to_md(self, table):
        """将表格转换为Markdown格式
        
        参数:
            table: 表格数据
            
        返回:
            str: Markdown格式的表格
        """
        # 提取表格单元格文本
        table_data = []
        for row in table['data']['grid']:
            table_row = [cell['text'] for cell in row]  # 提取每个单元格的文本
            table_data.append(table_row)
        
        # 检查表格是否有表头
        if len(table_data) > 1 and len(table_data[0]) > 0:
            try:
                # 使用第一行作为表头
                md_table = tabulate(
                    table_data[1:],  # 数据行
                    headers=table_data[0],  # 表头
                    tablefmt="github"  # GitHub风格的Markdown表格
                )
            except ValueError:
                # 如果转换失败，禁用数字解析
                md_table = tabulate(
                    table_data[1:],
                    headers=table_data[0],
                    tablefmt="github",
                    disable_numparse=True  # 禁用数字解析
                )
        else:
            # 如果表格没有表头
            md_table = tabulate(table_data, tablefmt="github")
        
        return md_table


    def assemble_pictures(self, data):
        """组装图片数据
        
        处理文档中的所有图片，生成结构化图片数据。
        
        参数:
            data: 文档数据
            
        返回:
            list: 组装好的图片列表
        """
        assembled_pictures = []  # 初始化图片列表
        
        # 处理每个图片
        for i, picture in enumerate(data['pictures']):
            # 处理图片块，获取子元素列表
            children_list = self._process_picture_block(picture, data)
            
            # 获取图片引用编号
            ref_num = picture['self_ref'].split('/')[-1]
            ref_num = int(ref_num)
            
            # 获取图片页码
            picture_page_num = picture['prov'][0]['page_no']
            # 获取图片边界框
            picture_bbox = picture['prov'][0]['bbox']
            # 转换边界框为列表格式
            picture_bbox = [
                picture_bbox['l'],  # 左
                picture_bbox['t'],  # 上
                picture_bbox['r'],  # 右
                picture_bbox['b']   # 下
            ]
            
            # 创建图片对象
            picture_obj = {
                'picture_id': ref_num,  # 图片ID
                'page': picture_page_num,  # 页码
                'bbox': picture_bbox,  # 边界框
                'children': children_list,  # 子元素列表
            }
            # 添加到图片列表
            assembled_pictures.append(picture_obj)
            
        return assembled_pictures


    def _process_picture_block(self, picture, data):
        """处理图片块
        
        提取图片块中的文本引用。
        
        参数:
            picture: 图片数据
            data: 文档数据
            
        返回:
            list: 处理后的子元素列表
        """
        children_list = []  # 初始化子元素列表
        
        # 处理图片中的每个项目
        for item in picture['children']:
            # 处理引用类型的项目
            if isinstance(item, dict) and '$ref' in item:
                ref = item['$ref']  # 获取引用路径
                ref_type, ref_num = ref.split('/')[-2:]  # 解析引用类型和编号
                ref_num = int(ref_num)  # 转换为整数
                
                # 处理文本引用
                if ref_type == 'texts':
                    # 处理文本引用，创建内容项
                    content_item = self._process_text_reference(ref_num, data)
                    # 添加到子元素列表
                    children_list.append(content_item)

        return children_list


    def export_to_markdown(self, reports_dir: Path, output_dir: Path):
        """将JSON报告导出为Markdown格式
        
        参数:
            reports_dir: JSON报告目录
            output_dir: Markdown输出目录
        """
        # 创建输出目录
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 处理每个JSON报告
        for report_path in reports_dir.glob("*.json"):
            # 读取JSON报告
            with open(report_path, 'r', encoding='utf-8') as f:
                report_data = json.load(f)
                
            # 处理报告数据
            processed_report = self.process_report(report_data)
            
            # 生成Markdown文本
            document_text = ""
            for page in processed_report['pages']:
                # 添加页面分隔符和标题
                document_text += f"\n\n---\n\n# 第 {page['page']} 页\n\n"
                # 添加页面文本
                document_text += page['text']
                
            # 使用SHA1作为Markdown文件名
            report_name = report_data['metainfo'].get('sha1', 'unknown')
            # 保存Markdown文件
            with open(output_dir / f"{report_name}.md", "w", encoding="utf-8") as f:
                f.write(document_text)


    def expand_groups(self, body_children, groups):
        """展开组引用
        
        将组引用展开为实际内容项，并添加组信息。
        
        参数:
            body_children: 主体子元素列表
            groups: 组列表
            
        返回:
            list: 展开后的子元素列表
        """
        expanded_children = []  # 初始化展开后的子元素列表

        # 处理每个子元素
        for item in body_children:
            # 处理引用类型的项目
            if isinstance(item, dict) and '$ref' in item:
                ref = item['$ref']  # 获取引用路径
                ref_type, ref_num = ref.split('/')[-2:]  # 解析引用类型和编号
                ref_num = int(ref_num)  # 转换为整数

                # 处理组引用
                if ref_type == 'groups':
                    group = groups[ref_num]  # 获取组
                    group_id = ref_num  # 组ID
                    group_name = group.get('name', '')  # 组名称
                    group_label = group.get('label', '')  # 组标签

                    # 处理组中的每个子元素
                    for child in group['children']:
                        # 复制子元素并添加组信息
                        child_copy = child.copy()
                        child_copy['group_id'] = group_id
                        child_copy['group_name'] = group_name
                        child_copy['group_label'] = group_label
                        # 添加到展开后的子元素列表
                        expanded_children.append(child_copy)
                else:
                    # 非组引用，直接添加
                    expanded_children.append(item)
            else:
                # 非引用项目，直接添加
                expanded_children.append(item)

        return expanded_children