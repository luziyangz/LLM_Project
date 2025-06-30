# -*- coding: utf-8 -*-
"""
TextSplitter 文本分块器测试脚本

测试 TextSplitter 类的各项功能，包括：
1. 基础初始化测试
2. Token 计数功能测试
3. Markdown 文件分块测试
4. 批量 Markdown 处理测试
5. 序列化表格处理测试
6. CSV 元数据处理测试
7. 错误处理测试
"""

import os
import sys
import json
import tempfile
import logging
import unittest
from pathlib import Path
from typing import List, Dict
import time
import pandas as pd

# 添加src目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from text_splitter import TextSplitter
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保src目录中存在text_splitter.py文件，并且已安装必要的依赖")
    sys.exit(1)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('test_text_splitter.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TestTextSplitter(unittest.TestCase):
    """TextSplitter 文本分块器单元测试类"""
    
    @classmethod
    def setUpClass(cls):
        """测试类初始化"""
        cls.test_output_dir = Path("../debug_data/test_output/text_splitter")
        cls.test_md_dir = Path("../debug_data/test_output/test_markdown")
        cls.csv_test_file = Path("../debug_data/test_text_splitter_metadata.csv")
        
        # 创建测试目录
        cls.test_output_dir.mkdir(parents=True, exist_ok=True)
        cls.test_md_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("TextSplitter 测试环境初始化完成")
    
    def setUp(self):
        """每个测试方法前的准备"""
        self.start_time = time.time()
        self.text_splitter = TextSplitter()
    
    def tearDown(self):
        """每个测试方法后的清理"""
        elapsed = time.time() - self.start_time
        logger.info(f"测试耗时: {elapsed:.2f}秒")
    
    def create_test_markdown_files(self) -> List[Path]:
        """创建测试用的 Markdown 文件"""
        test_files = []
        
        # 创建测试文件1 - 短文件
        md_content_1 = """# 测试报告 A

## 概述
这是一个测试用的 Markdown 文件。
包含多行内容用于测试分块功能。

## 详细信息
- 项目名称：测试项目A
- 状态：进行中
- 负责人：张三

### 技术栈
1. Python
2. FastAPI
3. PostgreSQL

## 结论
测试文件创建成功。
"""
        
        # 创建测试文件2 - 长文件
        md_content_2 = """# 中芯国际研究报告

## 执行摘要
中芯国际集成电路制造有限公司是中国大陆规模最大的集成电路晶圆代工企业。
公司提供0.35微米到14纳米不同技术节点的晶圆代工与技术服务。

## 公司概况
### 基本信息
- 公司名称：中芯国际集成电路制造有限公司
- 股票代码：688981.SH / 0981.HK
- 成立时间：2000年
- 总部地址：上海市浦东新区

### 业务范围
1. 集成电路晶圆代工服务
2. 集成电路设计服务
3. 技术开发服务
4. 设备销售

## 财务分析
### 营收情况
2023年全年营收达到63.2亿美元，同比增长4.3%。
其中，28纳米及以下先进工艺营收占比达到18.9%。

### 盈利能力
毛利率为19.5%，净利率为8.2%。
ROE为12.3%，ROA为6.8%。

## 技术发展
### 工艺节点
- 14纳米：已实现量产
- 12纳米：正在开发
- 7纳米：研发中

### 产能情况
月产能约55万片8英寸等效晶圆。
产能利用率维持在85%以上。

## 市场地位
在全球晶圆代工市场中排名第四。
在中国大陆市场占有率超过60%。

## 风险因素
1. 国际贸易摩擦影响
2. 技术升级压力
3. 市场竞争加剧
4. 原材料价格波动

## 投资建议
基于公司的技术实力和市场地位，给予"买入"评级。
目标价格：25元人民币。

## 附录
### 财务数据表
| 年份 | 营收(亿美元) | 净利润(亿美元) | 毛利率(%) |
|------|-------------|---------------|----------|
| 2021 | 54.4        | 16.9          | 25.8     |
| 2022 | 60.8        | 7.9           | 13.0     |
| 2023 | 63.2        | 5.2           | 19.5     |

### 技术路线图
- 2024年：12纳米量产
- 2025年：7纳米试产
- 2026年：5纳米研发
"""
        
        # 写入测试文件
        test_file_1 = self.test_md_dir / "test_report_a.md"
        test_file_2 = self.test_md_dir / "smic_research_report.md"
        
        with open(test_file_1, 'w', encoding='utf-8') as f:
            f.write(md_content_1)
        test_files.append(test_file_1)
        
        with open(test_file_2, 'w', encoding='utf-8') as f:
            f.write(md_content_2)
        test_files.append(test_file_2)
        
        logger.info(f"创建了 {len(test_files)} 个测试 Markdown 文件")
        return test_files
    
    def create_test_csv(self) -> Path:
        """创建测试用的CSV元数据文件"""
        csv_content = '''sha1,company_name,file_name
abc123def456,"测试公司A",test_report_a.md
789xyz012uvw,"测试公司B",test_report_b.md
test_hash_smic,"中芯国际",smic_research_report.md
test_hash_002,"上海证券",shanghai_securities_report.md
'''
        
        self.csv_test_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.csv_test_file, 'w', encoding='utf-8') as f:
            f.write(csv_content)
            
        logger.info(f"创建测试CSV文件: {self.csv_test_file}")
        return self.csv_test_file
    
    def create_test_serialized_tables(self) -> Path:
        """创建测试用的序列化表格文件"""
        tables_data = {
            "tables": [
                {
                    "table_id": "table_1",
                    "page": 1,
                    "serialized": {
                        "information_blocks": [
                            {"information_block": "财务数据表格第一行"},
                            {"information_block": "营收: 63.2亿美元"},
                            {"information_block": "净利润: 5.2亿美元"}
                        ]
                    }
                },
                {
                    "table_id": "table_2",
                    "page": 2,
                    "serialized": {
                        "information_blocks": [
                            {"information_block": "技术节点表格"},
                            {"information_block": "14纳米: 已量产"},
                            {"information_block": "12纳米: 开发中"}
                        ]
                    }
                }
            ]
        }
        
        tables_file = self.test_output_dir / "test_serialized_tables.json"
        with open(tables_file, 'w', encoding='utf-8') as f:
            json.dump(tables_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"创建测试序列化表格文件: {tables_file}")
        return tables_file
    
    def test_01_basic_initialization(self):
        """测试基础初始化功能"""
        logger.info("=== 测试基础初始化 ===")
        
        # 测试默认初始化
        splitter = TextSplitter()
        self.assertIsNotNone(splitter)
        logger.info("✓ 默认初始化成功")
    
    def test_02_token_counting(self):
        """测试Token计数功能"""
        logger.info("=== 测试Token计数功能 ===")
        
        # 测试基本文本
        test_text = "这是一个测试文本，用于验证token计数功能。"
        token_count = self.text_splitter.count_tokens(test_text)
        self.assertIsInstance(token_count, int)
        self.assertGreater(token_count, 0)
        logger.info(f"✓ 基本文本token计数: {token_count}")
        
        # 测试空文本
        empty_token_count = self.text_splitter.count_tokens("")
        self.assertEqual(empty_token_count, 0)
        logger.info("✓ 空文本token计数: 0")
        
        # 测试长文本
        long_text = "这是一个很长的测试文本。" * 100
        long_token_count = self.text_splitter.count_tokens(long_text)
        self.assertGreater(long_token_count, token_count)
        logger.info(f"✓ 长文本token计数: {long_token_count}")
    
    def test_03_markdown_file_splitting(self):
        """测试单个Markdown文件分块功能"""
        logger.info("=== 测试Markdown文件分块 ===")
        
        # 创建测试文件
        test_files = self.create_test_markdown_files()
        test_file = test_files[0]  # 使用第一个测试文件
        
        # 测试默认参数分块
        chunks = self.text_splitter.split_markdown_file(test_file)
        self.assertIsInstance(chunks, list)
        self.assertGreater(len(chunks), 0)
        
        # 验证分块结构
        for chunk in chunks:
            self.assertIn('lines', chunk)
            self.assertIn('text', chunk)
            self.assertIsInstance(chunk['lines'], list)
            self.assertEqual(len(chunk['lines']), 2)  # [start, end]
            self.assertIsInstance(chunk['text'], str)
        
        logger.info(f"✓ 默认参数分块成功，共 {len(chunks)} 个分块")
        
        # 测试自定义参数分块
        custom_chunks = self.text_splitter.split_markdown_file(
            test_file, chunk_size=5, chunk_overlap=2
        )
        self.assertGreater(len(custom_chunks), len(chunks))
        logger.info(f"✓ 自定义参数分块成功，共 {len(custom_chunks)} 个分块")
    
    def test_04_batch_markdown_processing(self):
        """测试批量Markdown处理功能"""
        logger.info("=== 测试批量Markdown处理 ===")
        
        # 创建测试文件
        test_files = self.create_test_markdown_files()
        
        # 测试批量处理
        output_dir = self.test_output_dir / "batch_output"
        self.text_splitter.split_markdown_reports(
            all_md_dir=self.test_md_dir,
            output_dir=output_dir,
            chunk_size=10,
            chunk_overlap=2
        )
        
        # 验证输出文件
        output_files = list(output_dir.glob("*.json"))
        self.assertEqual(len(output_files), len(test_files))
        
        # 验证输出文件内容
        for output_file in output_files:
            with open(output_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.assertIn('metainfo', data)
            self.assertIn('content', data)
            self.assertIn('chunks', data['content'])
            self.assertIsInstance(data['content']['chunks'], list)
            
            # 验证metainfo结构
            metainfo = data['metainfo']
            self.assertIn('sha1', metainfo)
            self.assertIn('company_name', metainfo)
            self.assertIn('file_name', metainfo)
        
        logger.info(f"✓ 批量处理成功，生成 {len(output_files)} 个JSON文件")
    
    def test_05_csv_metadata_integration(self):
        """测试CSV元数据集成功能"""
        logger.info("=== 测试CSV元数据集成 ===")
        
        # 创建测试文件和CSV
        test_files = self.create_test_markdown_files()
        csv_file = self.create_test_csv()
        
        # 测试带CSV的批量处理
        output_dir = self.test_output_dir / "csv_output"
        self.text_splitter.split_markdown_reports(
            all_md_dir=self.test_md_dir,
            output_dir=output_dir,
            chunk_size=15,
            chunk_overlap=3,
            subset_csv=csv_file
        )
        
        # 验证元数据是否正确填充
        for output_file in output_dir.glob("*.json"):
            with open(output_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            metainfo = data['metainfo']
            file_name = metainfo['file_name']
            
            # 检查是否有对应的公司名称
            if file_name in ['test_report_a.md', 'smic_research_report.md']:
                self.assertNotEqual(metainfo['company_name'], "")
                logger.info(f"✓ 文件 {file_name} 的公司名称: {metainfo['company_name']}")
        
        logger.info("✓ CSV元数据集成测试成功")
    
    def test_06_serialized_tables_processing(self):
        """测试序列化表格处理功能"""
        logger.info("=== 测试序列化表格处理 ===")
        
        # 创建测试数据
        tables_file = self.create_test_serialized_tables()
        
        # 加载表格数据
        with open(tables_file, 'r', encoding='utf-8') as f:
            tables_data = json.load(f)
        
        # 测试按页分组功能
        tables_by_page = self.text_splitter._get_serialized_tables_by_page(
            tables_data['tables']
        )
        
        self.assertIsInstance(tables_by_page, dict)
        self.assertIn(1, tables_by_page)
        self.assertIn(2, tables_by_page)
        
        # 验证表格数据结构
        for page, tables in tables_by_page.items():
            self.assertIsInstance(tables, list)
            for table in tables:
                self.assertIn('page', table)
                self.assertIn('text', table)
                self.assertIn('table_id', table)
                self.assertIn('length_tokens', table)
                self.assertIsInstance(table['length_tokens'], int)
        
        logger.info(f"✓ 序列化表格处理成功，共 {len(tables_by_page)} 页表格")
    
    def test_07_error_handling(self):
        """测试错误处理功能"""
        logger.info("=== 测试错误处理 ===")
        
        # 测试不存在的文件
        non_existent_file = Path("non_existent_file.md")
        with self.assertRaises(FileNotFoundError):
            self.text_splitter.split_markdown_file(non_existent_file)
        logger.info("✓ 不存在文件的错误处理正常")
        
        # 测试不存在的目录
        non_existent_dir = Path("non_existent_directory")
        output_dir = self.test_output_dir / "error_test"
        
        # 这应该不会抛出异常，因为方法会创建输出目录
        try:
            self.text_splitter.split_markdown_reports(
                all_md_dir=non_existent_dir,
                output_dir=output_dir
            )
            logger.info("✓ 不存在目录的处理正常（无文件处理）")
        except Exception as e:
            logger.info(f"✓ 不存在目录抛出预期异常: {e}")
    
    def test_08_performance_benchmark(self):
        """测试性能基准"""
        logger.info("=== 测试性能基准 ===")
        
        # 创建大量测试文件
        large_content = "# 大文件测试\n\n" + "这是测试内容。\n" * 1000
        large_file = self.test_md_dir / "large_test_file.md"
        
        with open(large_file, 'w', encoding='utf-8') as f:
            f.write(large_content)
        
        # 测试处理时间
        start_time = time.time()
        chunks = self.text_splitter.split_markdown_file(large_file, chunk_size=50)
        processing_time = time.time() - start_time
        
        self.assertGreater(len(chunks), 0)
        logger.info(f"✓ 大文件处理完成，耗时: {processing_time:.3f}秒，生成 {len(chunks)} 个分块")
        
        # 测试token计数性能
        start_time = time.time()
        token_count = self.text_splitter.count_tokens(large_content)
        token_time = time.time() - start_time
        
        logger.info(f"✓ Token计数完成，耗时: {token_time:.3f}秒，共 {token_count} 个token")

def run_all_tests():
    """运行所有测试"""
    logger.info("开始运行 TextSplitter 测试套件")
    
    # 创建测试套件
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestTextSplitter)
    
    # 运行测试
    runner = unittest.TextTestRunner(
        verbosity=2,
        stream=sys.stdout,
        buffer=True
    )
    
    result = runner.run(test_suite)
    
    # 输出测试结果摘要
    logger.info(f"\n{'='*50}")
    logger.info("测试结果摘要:")
    logger.info(f"总测试数: {result.testsRun}")
    logger.info(f"成功: {result.testsRun - len(result.failures) - len(result.errors)}")
    logger.info(f"失败: {len(result.failures)}")
    logger.info(f"错误: {len(result.errors)}")
    
    if result.failures:
        logger.error("失败的测试:")
        for test, traceback in result.failures:
            logger.error(f"  - {test}: {traceback}")
    
    if result.errors:
        logger.error("错误的测试:")
        for test, traceback in result.errors:
            logger.error(f"  - {test}: {traceback}")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)