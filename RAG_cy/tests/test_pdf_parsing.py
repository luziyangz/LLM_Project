#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDF解析器测试脚本

测试PDFParser类的各项功能，包括：
1. 基础初始化测试
2. CSV元数据解析测试
3. 单个PDF文件解析测试
4. 批量PDF文件解析测试
5. 错误处理测试
6. 性能基准测试
"""

import os
import sys
import json
import tempfile
import logging
import unittest
from pathlib import Path
from typing import List
import time

# 添加src目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from pdf_parsing import PDFParser, JsonReportProcessor
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保src目录中存在pdf_parsing.py文件")
    sys.exit(1)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('test_pdf_parsing.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TestPDFParser(unittest.TestCase):
    """PDF解析器单元测试类"""
    
    @classmethod
    def setUpClass(cls):
        """测试类初始化"""
        cls.test_data_dir = Path("../data/pdf_reports")  # PDF测试文件目录
        cls.test_output_dir = Path("../debug_data/test_output")  # 测试输出目录
        cls.csv_test_file = Path("../debug_data/test_metadata.csv")  # 测试CSV文件
        
        # 创建测试目录
        cls.test_output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("测试环境初始化完成")
    
    def setUp(self):
        """每个测试方法前的准备"""
        self.start_time = time.time()
    
    def tearDown(self):
        """每个测试方法后的清理"""
        elapsed = time.time() - self.start_time
        logger.info(f"测试耗时: {elapsed:.2f}秒")
    
    def create_test_csv(self) -> Path:
        """创建测试用的CSV元数据文件"""
        csv_content = '''sha1,company_name,file_name
abc123def456,"测试公司A",test_report_a.pdf
789xyz012uvw,"测试公司B",test_report_b.pdf
test_hash_001,"中芯国际",中芯国际机构调研纪要.pdf
test_hash_002,"上海证券",上海证券研究报告.pdf
'''
        
        self.csv_test_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.csv_test_file, 'w', encoding='utf-8') as f:
            f.write(csv_content)
            
        logger.info(f"创建测试CSV文件: {self.csv_test_file}")
        return self.csv_test_file
    
    def test_01_basic_initialization(self):
        """测试基础初始化功能"""
        logger.info("=== 测试基础初始化 ===")
        
        # 测试默认初始化
        parser1 = PDFParser()
        self.assertIsNotNone(parser1.doc_converter)
        self.assertEqual(parser1.metadata_lookup, {})
        logger.info("✓ 默认初始化成功")
        
        # 测试带参数初始化
        parser2 = PDFParser(
            output_dir=self.test_output_dir,
            num_threads=2
        )
        self.assertEqual(parser2.output_dir, self.test_output_dir)
        self.assertEqual(parser2.num_threads, 2)
        logger.info("✓ 带参数初始化成功")
        
        # 测试带CSV元数据初始化
        csv_file = self.create_test_csv()
        parser3 = PDFParser(
            output_dir=self.test_output_dir,
            csv_metadata_path=csv_file
        )
        self.assertGreater(len(parser3.metadata_lookup), 0)
        logger.info(f"✓ 带CSV元数据初始化成功，元数据条目数: {len(parser3.metadata_lookup)}")
    
    def test_02_csv_metadata_parsing(self):
        """测试CSV元数据解析功能"""
        logger.info("=== 测试CSV元数据解析 ===")
        
        csv_file = self.create_test_csv()
        metadata = PDFParser._parse_csv_metadata(csv_file)
        
        # 验证解析结果
        expected_keys = ['abc123def456', '789xyz012uvw', 'test_hash_001', 'test_hash_002']
        for key in expected_keys:
            self.assertIn(key, metadata)
            self.assertIn('company_name', metadata[key])
            logger.info(f"✓ 找到元数据: {key} -> {metadata[key]['company_name']}")
        
        # 测试公司名称去引号功能
        self.assertEqual(metadata['abc123def456']['company_name'], '测试公司A')
        self.assertEqual(metadata['test_hash_001']['company_name'], '中芯国际')
        logger.info("✓ 公司名称去引号功能正常")
    
    def test_03_pdf_file_detection(self):
        """测试PDF文件检测功能"""
        logger.info("=== 测试PDF文件检测 ===")
        
        if not self.test_data_dir.exists():
            self.skipTest(f"PDF测试目录不存在: {self.test_data_dir}")
        
        # 检测PDF文件
        pdf_files = list(self.test_data_dir.glob("*.pdf"))
        self.assertGreater(len(pdf_files), 0, "应该至少有一个PDF文件")
        
        logger.info(f"检测到 {len(pdf_files)} 个PDF文件:")
        for pdf_file in pdf_files:
            file_size = pdf_file.stat().st_size
            self.assertGreater(file_size, 0, f"文件 {pdf_file.name} 不应为空")
            logger.info(f"  - {pdf_file.name} ({file_size} bytes)")
    
    def test_04_single_pdf_parsing(self):
        """测试单个PDF文件解析"""
        logger.info("=== 测试单个PDF文件解析 ===")
        
        # 获取第一个PDF文件
        pdf_files = list(self.test_data_dir.glob("*.pdf"))
        if not pdf_files:
            self.skipTest("没有找到PDF文件进行测试")
        
        test_pdf = pdf_files[0]
        logger.info(f"测试文件: {test_pdf.name}")
        
        # 创建解析器
        parser = PDFParser(
            output_dir=self.test_output_dir / "single_test",
            num_threads=1
        )
        
        # 解析单个文件
        result = parser.parse_and_export(input_doc_paths=[test_pdf])
        
        # 验证结果
        self.assertIsInstance(result, dict)
        self.assertIn('total', result)
        self.assertIn('success', result)
        self.assertIn('failed', result)
        self.assertEqual(result['total'], 1)
        
        logger.info(f"解析结果: {result}")
        
        if result['success'] > 0:
            # 检查输出文件
            output_file = parser.output_dir / f"{test_pdf.stem}.json"
            self.assertTrue(output_file.exists(), "输出文件应该存在")
            
            # 验证JSON格式
            with open(output_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.assertIsInstance(data, dict)
                expected_keys = ['metainfo', 'content', 'tables', 'pictures']
                for key in expected_keys:
                    self.assertIn(key, data, f"输出JSON应包含 {key} 字段")
                
            logger.info(f"✓ JSON格式正确，包含字段: {list(data.keys())}")
    
    def test_05_batch_pdf_parsing(self):
        """测试批量PDF文件解析"""
        logger.info("=== 测试批量PDF文件解析 ===")
        
        if not self.test_data_dir.exists():
            self.skipTest(f"PDF测试目录不存在: {self.test_data_dir}")
        
        # 创建带元数据的解析器
        csv_file = self.create_test_csv()
        parser = PDFParser(
            output_dir=self.test_output_dir / "batch_test",
            csv_metadata_path=csv_file,
            num_threads=2
        )
        
        # 批量解析目录中的前3个PDF（避免测试时间过长）
        pdf_files = list(self.test_data_dir.glob("*.pdf"))[:3]
        result = parser.parse_and_export(input_doc_paths=pdf_files)
        
        # 验证结果
        self.assertGreater(result['total'], 0)
        self.assertIsInstance(result['elapsed_time'], (int, float))
        
        logger.info(f"批量解析结果: {result}")
        logger.info(f"✓ 批量解析完成，处理了 {result['total']} 个文件")
        logger.info(f"  成功: {result['success']} 个")
        logger.info(f"  失败: {result['failed']} 个")
        logger.info(f"  耗时: {result['elapsed_time']:.2f} 秒")
    
    def test_06_error_handling(self):
        """测试错误处理功能"""
        logger.info("=== 测试错误处理 ===")
        
        parser = PDFParser(output_dir=self.test_output_dir / "error_test")
        
        # 测试不存在的文件
        non_existent_file = Path("non_existent_file.pdf")
        result = parser.parse_and_export(input_doc_paths=[non_existent_file])
        
        self.assertEqual(result['total'], 1)
        self.assertEqual(result['failed'], 1)
        self.assertEqual(result['success'], 0)
        logger.info("✓ 错误处理正常，正确识别了不存在的文件")
        
        # 测试无效参数
        with self.assertRaises(ValueError):
            parser.parse_and_export()  # 没有提供任何参数
        logger.info("✓ 参数验证正常")
    
    def test_07_json_report_processor(self):
        """测试JSON报告处理器"""
        logger.info("=== 测试JSON报告处理器 ===")
        
        # 创建测试元数据
        csv_file = self.create_test_csv()
        metadata = PDFParser._parse_csv_metadata(csv_file)
        
        # 创建处理器
        processor = JsonReportProcessor(
            metadata_lookup=metadata,
            debug_data_path=self.test_output_dir / "debug"
        )
        
        self.assertEqual(processor.metadata_lookup, metadata)
        self.assertIsNotNone(processor.debug_data_path)
        logger.info("✓ JSON报告处理器初始化成功")

class PDFParserIntegrationTester:
    """PDF解析器集成测试类（非单元测试）"""
    
    def __init__(self):
        """初始化集成测试环境"""
        self.test_data_dir = Path("../data/pdf_reports")
        self.test_output_dir = Path("../debug_data/integration_test")
        self.csv_test_file = Path("../debug_data/test_metadata.csv")
        
        # 创建测试目录
        self.test_output_dir.mkdir(parents=True, exist_ok=True)
    
    def run_performance_test(self, max_files: int = 5):
        """运行性能测试"""
        logger.info("=== 性能基准测试 ===")
        
        if not self.test_data_dir.exists():
            logger.warning(f"PDF测试目录不存在: {self.test_data_dir}")
            return
        
        pdf_files = list(self.test_data_dir.glob("*.pdf"))[:max_files]
        if not pdf_files:
            logger.warning("没有找到PDF文件进行性能测试")
            return
        
        # 单线程测试
        parser_single = PDFParser(
            output_dir=self.test_output_dir / "perf_single",
            num_threads=1
        )
        
        start_time = time.time()
        result_single = parser_single.parse_and_export(input_doc_paths=pdf_files)
        single_time = time.time() - start_time
        
        # 多线程测试
        parser_multi = PDFParser(
            output_dir=self.test_output_dir / "perf_multi",
            num_threads=4
        )
        
        start_time = time.time()
        result_multi = parser_multi.parse_and_export(input_doc_paths=pdf_files)
        multi_time = time.time() - start_time
        
        # 输出性能对比
        logger.info(f"性能测试结果 (处理 {len(pdf_files)} 个文件):")
        logger.info(f"  单线程: {single_time:.2f}秒, 成功率: {result_single['success']}/{result_single['total']}")
        logger.info(f"  多线程: {multi_time:.2f}秒, 成功率: {result_multi['success']}/{result_multi['total']}")
        
        if single_time > 0:
            speedup = single_time / multi_time
            logger.info(f"  加速比: {speedup:.2f}x")
        
        return {
            'single_thread': {'time': single_time, 'result': result_single},
            'multi_thread': {'time': multi_time, 'result': result_multi}
        }
    
    def run_full_integration_test(self):
        """运行完整集成测试"""
        logger.info("=== 完整集成测试 ===")
        
        # 创建测试CSV
        csv_content = '''sha1,company_name,file_name
real_hash_001,"中芯国际",中芯国际机构调研纪要.pdf
real_hash_002,"上海证券",上海证券研究报告.pdf
'''
        
        self.csv_test_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.csv_test_file, 'w', encoding='utf-8') as f:
            f.write(csv_content)
        
        # 创建完整配置的解析器
        parser = PDFParser(
            output_dir=self.test_output_dir / "full_test",
            csv_metadata_path=self.csv_test_file,
            num_threads=2
        )
        parser.debug_data_path = self.test_output_dir / "debug"
        
        # 运行完整解析流程
        if self.test_data_dir.exists():
            result = parser.parse_and_export(doc_dir=self.test_data_dir)
            
            logger.info(f"完整集成测试结果: {result}")
            
            # 验证输出文件
            output_files = list((self.test_output_dir / "full_test").glob("*.json"))
            logger.info(f"生成了 {len(output_files)} 个输出文件")
            
            # 检查第一个输出文件的结构
            if output_files:
                with open(output_files[0], 'r', encoding='utf-8') as f:
                    sample_data = json.load(f)
                    logger.info(f"示例输出结构: {list(sample_data.keys())}")
                    
                    if 'metainfo' in sample_data:
                        logger.info(f"元数据字段: {list(sample_data['metainfo'].keys())}")
            
            return result
        else:
            logger.warning(f"PDF测试目录不存在: {self.test_data_dir}")
            return None

def run_unit_tests():
    """运行单元测试"""
    logger.info("开始运行PDF解析器单元测试")
    logger.info("=" * 60)
    
    # 创建测试套件
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestPDFParser)
    
    # 运行测试
    runner = unittest.TextTestRunner(
        verbosity=2,
        stream=sys.stdout,
        buffer=True
    )
    
    result = runner.run(test_suite)
    
    # 输出测试总结
    logger.info("=" * 60)
    logger.info(f"单元测试完成: 运行 {result.testsRun} 个测试")
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

def run_integration_tests():
    """运行集成测试"""
    logger.info("开始运行PDF解析器集成测试")
    logger.info("=" * 60)
    
    tester = PDFParserIntegrationTester()
    
    try:
        # 运行性能测试
        perf_result = tester.run_performance_test(max_files=3)
        
        # 运行完整集成测试
        integration_result = tester.run_full_integration_test()
        
        logger.info("=" * 60)
        logger.info("集成测试完成")
        
        return True
        
    except Exception as e:
        logger.error(f"集成测试失败: {str(e)}")
        return False

def main():
    """主函数"""
    print("PDF解析器测试套件")
    print("=" * 60)
    
    # 检查依赖
    try:
        from docling.document_converter import DocumentConverter
        print("✓ Docling库已安装")
    except ImportError:
        print("✗ Docling库未安装，请先安装: pip install docling")
        return False
    
    # 运行单元测试
    unit_success = run_unit_tests()
    
    # 运行集成测试
    integration_success = run_integration_tests()
    
    # 总结
    print("\n" + "=" * 60)
    print("测试总结:")
    print(f"  单元测试: {'✓ 通过' if unit_success else '✗ 失败'}")
    print(f"  集成测试: {'✓ 通过' if integration_success else '✗ 失败'}")
    
    if unit_success and integration_success:
        print("\n🎉 所有测试都通过了！")
        return True
    else:
        print("\n⚠️  部分测试失败，请检查日志")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)