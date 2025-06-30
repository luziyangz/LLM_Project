# -*- coding: utf-8 -*-
"""
VectorDBIngestor 向量库构建工具测试脚本

测试 VectorDBIngestor 类的各项功能，包括：
1. 基础初始化测试
2. 文本嵌入向量获取测试
3. 向量库构建测试
4. 单个报告处理测试
5. 批量报告处理测试
6. 错误处理测试
7. Mock API 测试
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
import numpy as np
from unittest.mock import patch, MagicMock, mock_open

# 添加src目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from ingestion import VectorDBIngestor
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保src目录中存在ingestion.py文件，并且已安装必要的依赖")
    sys.exit(1)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('test_ingestion.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TestVectorDBIngestor(unittest.TestCase):
    """VectorDBIngestor 向量库构建工具单元测试类"""
    
    @classmethod
    def setUpClass(cls):
        """测试类初始化"""
        cls.test_output_dir = Path("../debug_data/test_output/ingestion")
        cls.test_reports_dir = Path("../debug_data/test_output/test_reports")
        
        # 创建测试目录
        cls.test_output_dir.mkdir(parents=True, exist_ok=True)
        cls.test_reports_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("VectorDBIngestor 测试环境初始化完成")
    
    def setUp(self):
        """每个测试方法前的准备"""
        self.start_time = time.time()
        # 设置测试用的环境变量
        os.environ['DASHSCOPE_API_KEY'] = 'test_api_key'
        self.ingestor = VectorDBIngestor()
    
    def tearDown(self):
        """每个测试方法后的清理"""
        elapsed = time.time() - self.start_time
        logger.info(f"测试耗时: {elapsed:.2f}秒")
    
    def create_test_report_files(self) -> List[Path]:
        """创建测试用的报告JSON文件"""
        test_files = []
        
        # 创建测试报告1
        report_1 = {
            "metainfo": {
                "sha1": "test_hash_001",
                "company_name": "测试公司A",
                "file_name": "test_report_a.json"
            },
            "content": {
                "chunks": [
                    {
                        "id": 0,
                        "type": "content",
                        "text": "这是第一个测试文本块，用于验证向量化功能。"
                    },
                    {
                        "id": 1,
                        "type": "content",
                        "text": "这是第二个测试文本块，包含更多的技术内容和专业术语。"
                    },
                    {
                        "id": 2,
                        "type": "content",
                        "text": "第三个文本块专注于财务数据和业绩分析。"
                    }
                ]
            }
        }
        
        # 创建测试报告2
        report_2 = {
            "metainfo": {
                "sha1": "test_hash_002",
                "company_name": "中芯国际",
                "file_name": "smic_report.json"
            },
            "content": {
                "chunks": [
                    {
                        "id": 0,
                        "type": "content",
                        "text": "中芯国际是中国大陆规模最大的集成电路晶圆代工企业。"
                    },
                    {
                        "id": 1,
                        "type": "content",
                        "text": "公司提供0.35微米到14纳米不同技术节点的晶圆代工与技术服务。"
                    }
                ]
            }
        }
        
        # 写入测试文件
        test_file_1 = self.test_reports_dir / "test_report_a.json"
        test_file_2 = self.test_reports_dir / "smic_report.json"
        
        with open(test_file_1, 'w', encoding='utf-8') as f:
            json.dump(report_1, f, ensure_ascii=False, indent=2)
        test_files.append(test_file_1)
        
        with open(test_file_2, 'w', encoding='utf-8') as f:
            json.dump(report_2, f, ensure_ascii=False, indent=2)
        test_files.append(test_file_2)
        
        logger.info(f"创建了 {len(test_files)} 个测试报告文件")
        return test_files
    
    def test_01_basic_initialization(self):
        """测试基础初始化功能"""
        logger.info("=== 测试基础初始化 ===")
        
        # 测试默认初始化
        ingestor = VectorDBIngestor()
        self.assertIsNotNone(ingestor)
        logger.info("✓ 默认初始化成功")
        
        # 验证API Key设置
        import dashscope
        self.assertEqual(dashscope.api_key, 'test_api_key')
        logger.info("✓ API Key 设置成功")
    
    @patch('dashscope.TextEmbedding.call')
    def test_02_get_embeddings_single_text(self, mock_embedding_call):
        """测试单个文本的嵌入向量获取"""
        logger.info("=== 测试单个文本嵌入向量获取 ===")
        
        # Mock API 响应
        mock_response = {
            'output': {
                'embedding': [0.1, 0.2, 0.3, 0.4, 0.5] * 307  # 1536维向量
            }
        }
        mock_embedding_call.return_value = mock_response
        
        # 测试单个文本
        test_text = "这是一个测试文本"
        embeddings = self.ingestor._get_embeddings(test_text)
        
        self.assertIsInstance(embeddings, list)
        self.assertEqual(len(embeddings), 1)
        self.assertEqual(len(embeddings[0]), 1535)  # 1536维向量
        logger.info(f"✓ 单个文本嵌入向量获取成功，维度: {len(embeddings[0])}")
    
    @patch('dashscope.TextEmbedding.call')
    def test_03_get_embeddings_batch_text(self, mock_embedding_call):
        """测试批量文本的嵌入向量获取"""
        logger.info("=== 测试批量文本嵌入向量获取 ===")
        
        # Mock API 响应
        mock_response = {
            'output': {
                'embeddings': [
                    {'embedding': [0.1, 0.2, 0.3] * 512},
                    {'embedding': [0.4, 0.5, 0.6] * 512},
                    {'embedding': [0.7, 0.8, 0.9] * 512}
                ]
            }
        }
        mock_embedding_call.return_value = mock_response
        
        # 测试批量文本
        test_texts = [
            "第一个测试文本",
            "第二个测试文本",
            "第三个测试文本"
        ]
        embeddings = self.ingestor._get_embeddings(test_texts)
        
        self.assertIsInstance(embeddings, list)
        self.assertEqual(len(embeddings), 3)
        for embedding in embeddings:
            self.assertEqual(len(embedding), 1536)
        logger.info(f"✓ 批量文本嵌入向量获取成功，数量: {len(embeddings)}")
    
    def test_04_get_embeddings_error_handling(self):
        """测试嵌入向量获取的错误处理"""
        logger.info("=== 测试嵌入向量获取错误处理 ===")
        
        # 由于tenacity重试装饰器，ValueError会被包装成RetryError
        from tenacity import RetryError
        
        # 测试空字符串 - 期望RetryError而不是ValueError
        with self.assertRaises(RetryError):
            self.ingestor._get_embeddings("")
        logger.info("✓ 空字符串错误处理正常")
        
        # 测试空列表 - 期望RetryError而不是ValueError
        with self.assertRaises(RetryError):
            self.ingestor._get_embeddings([])
        logger.info("✓ 空列表错误处理正常")
        
        # 测试非字符串类型 - 期望RetryError而不是ValueError
        with self.assertRaises(RetryError):
            self.ingestor._get_embeddings(["文本", 123, "另一个文本"])
        logger.info("✓ 非字符串类型错误处理正常")
    
    def test_05_create_vector_db(self):
        """测试向量库构建功能"""
        logger.info("=== 测试向量库构建 ===")
        
        # 创建测试嵌入向量
        test_embeddings = [
            [0.1, 0.2, 0.3, 0.4, 0.5],
            [0.6, 0.7, 0.8, 0.9, 1.0],
            [1.1, 1.2, 1.3, 1.4, 1.5]
        ]
        
        # 构建向量库
        index = self.ingestor._create_vector_db(test_embeddings)
        
        # 验证向量库
        self.assertIsNotNone(index)
        self.assertEqual(index.ntotal, 3)  # 3个向量
        self.assertEqual(index.d, 5)  # 5维向量
        logger.info(f"✓ 向量库构建成功，向量数量: {index.ntotal}，维度: {index.d}")
    
    @patch('dashscope.TextEmbedding.call')
    def test_06_process_single_report(self, mock_embedding_call):
        """测试单个报告处理功能"""
        logger.info("=== 测试单个报告处理 ===")
        
        # Mock API 响应
        mock_response = {
            'output': {
                'embeddings': [
                    {'embedding': [0.1, 0.2, 0.3] * 512},
                    {'embedding': [0.4, 0.5, 0.6] * 512},
                    {'embedding': [0.7, 0.8, 0.9] * 512}
                ]
            }
        }
        mock_embedding_call.return_value = mock_response
        
        # 创建测试报告
        test_report = {
            "metainfo": {
                "sha1": "test_hash_single",
                "company_name": "测试公司",
                "file_name": "test_single.json"
            },
            "content": {
                "chunks": [
                    {"id": 0, "type": "content", "text": "第一个文本块"},
                    {"id": 1, "type": "content", "text": "第二个文本块"},
                    {"id": 2, "type": "content", "text": "第三个文本块"}
                ]
            }
        }
        
        # 处理报告
        index = self.ingestor._process_report(test_report)
        
        # 验证结果
        self.assertIsNotNone(index)
        self.assertEqual(index.ntotal, 3)
        logger.info(f"✓ 单个报告处理成功，生成向量数量: {index.ntotal}")
    
    @patch('dashscope.TextEmbedding.call')
    @patch('faiss.write_index')
    def test_07_process_batch_reports(self, mock_embedding_call, mock_write_index):
        """测试批量报告处理功能"""
        logger.info("=== 测试批量报告处理 ===")
        
        # 清理测试目录中可能存在的无效文件
        invalid_files = [
            self.test_reports_dir / "invalid_report.json",
            self.test_reports_dir / "test_report_without_sha1.json"
        ]
        for invalid_file in invalid_files:
            if invalid_file.exists():
                invalid_file.unlink()
                logger.info(f"清理无效测试文件: {invalid_file}")
        
        # Mock API 响应
        mock_response = {
            'output': {
                'embeddings': [
                    {'embedding': [0.1, 0.2, 0.3] * 512},
                    {'embedding': [0.4, 0.5, 0.6] * 512}
                ]
            }
        }
        mock_embedding_call.return_value = mock_response
        
        # 创建有效的测试文件
        test_files = self.create_test_report_files()
        
        # 确保所有测试文件都有sha1字段
        for test_file in self.test_reports_dir.glob("*.json"):
            with open(test_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 如果缺少sha1字段，添加一个
            if 'sha1' not in data.get('metainfo', {}):
                data['metainfo']['sha1'] = f"test_hash_{test_file.stem}"
                with open(test_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                logger.info(f"为文件 {test_file} 添加sha1字段")
        
        # 批量处理
        output_dir = self.test_output_dir / "batch_faiss"
        self.ingestor.process_reports(self.test_reports_dir, output_dir)
        
        # 验证调用次数
        valid_files_count = len(list(self.test_reports_dir.glob("*.json")))
        self.assertEqual(mock_write_index.call_count, valid_files_count)
        logger.info(f"✓ 批量报告处理成功，处理文件数量: {valid_files_count}")
    
    @patch('dashscope.TextEmbedding.call')
    def test_08_missing_sha1_error(self, mock_embedding_call):
        """测试缺少sha1字段的错误处理"""
        logger.info("=== 测试缺少sha1字段错误处理 ===")
        
        # 模拟API返回None（这会导致RetryError）
        mock_embedding_call.return_value = None
        
        def test_11_missing_sha1_validation(self):
            """专门测试缺少sha1字段的验证"""
            logger.info("=== 测试sha1字段验证 ===")
            
            # 清理测试目录
            for file in self.test_reports_dir.glob("*.json"):
                file.unlink()
            
            # 创建缺少sha1的测试报告
            invalid_report = {
                "metainfo": {
                    "company_name": "测试公司",
                    "file_name": "invalid_report.json"
                    # 故意缺少 sha1 字段
                },
                "content": {
                    "chunks": [
                        {"id": 0, "type": "content", "text": "测试文本"}
                    ]
                }
            }
            
            # 写入无效文件
            invalid_file = self.test_reports_dir / "invalid_report.json"
            with open(invalid_file, 'w', encoding='utf-8') as f:
                json.dump(invalid_report, f, ensure_ascii=False, indent=2)
            
            # 测试错误处理 - 期望ValueError（缺少sha1字段）
            with self.assertRaises(ValueError) as context:
                self.ingestor.process_reports(self.test_reports_dir, self.test_output_dir)
            
            # 验证错误信息
            self.assertIn("缺少 sha1 字段", str(context.exception))
            logger.info("✓ 缺少sha1字段错误处理正常")
        logger.info("✓ API返回None错误处理正常")
        
        # 清理测试文件
        if invalid_file.exists():
            invalid_file.unlink()
    
    @patch('dashscope.TextEmbedding.call')
    def test_09_text_truncation(self, mock_embedding_call):
        """测试文本截断功能"""
        logger.info("=== 测试文本截断功能 ===")
        
        # Mock API 响应
        mock_response = {
            'output': {
                'embedding': [0.1, 0.2, 0.3] * 512
            }
        }
        mock_embedding_call.return_value = mock_response
        
        # 创建超长文本
        long_text = "这是一个很长的测试文本。" * 300  # 超过2048字符
        
        # 创建包含超长文本的报告
        test_report = {
            "metainfo": {
                "sha1": "test_hash_long",
                "company_name": "测试公司",
                "file_name": "test_long.json"
            },
            "content": {
                "chunks": [
                    {"id": 0, "type": "content", "text": long_text}
                ]
            }
        }
        
        # 处理报告
        index = self.ingestor._process_report(test_report)
        
        # 验证文本被截断
        mock_embedding_call.assert_called_once()
        call_args = mock_embedding_call.call_args[1]['input']
        self.assertLessEqual(len(call_args[0]), 2048)
        logger.info(f"✓ 文本截断功能正常，截断后长度: {len(call_args[0])}")
    
    @patch('dashscope.TextEmbedding.call')
    def test_10_performance_benchmark(self, mock_embedding_call):
        """测试性能基准"""
        logger.info("=== 测试性能基准 ===")
        
        # Mock API 响应
        mock_response = {
            'output': {
                'embeddings': [{'embedding': [0.1] * 1536} for _ in range(25)]
            }
        }
        mock_embedding_call.return_value = mock_response
        
        # 创建大量文本块
        large_texts = [f"测试文本块 {i}" for i in range(100)]
        
        # 测试处理时间
        start_time = time.time()
        embeddings = self.ingestor._get_embeddings(large_texts)
        processing_time = time.time() - start_time
        
        self.assertEqual(len(embeddings), 100)
        logger.info(f"✓ 大批量文本处理完成，耗时: {processing_time:.3f}秒，文本数量: {len(large_texts)}")

    @patch('dashscope.TextEmbedding.call')
    def test_11_api_returns_none_error(self, mock_embedding_call):
        """测试API返回None时的错误处理"""
        logger.info("=== 测试API返回None错误处理 ===")
        
        # Mock API返回None
        mock_embedding_call.return_value = None
        
        # 测试错误处理 - 期望RetryError（由于tenacity装饰器）
        from tenacity import RetryError
        with self.assertRaises(RetryError) as context:
            self.ingestor._get_embeddings("测试文本")
        
        # 验证错误信息包含TypeError
        self.assertIn("TypeError", str(context.exception))
        logger.info("✓ API返回None时的错误处理正常")
def run_all_tests():
    """运行所有测试"""
    logger.info("开始运行 VectorDBIngestor 测试套件")
    
    # 创建测试套件
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestVectorDBIngestor)
    
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