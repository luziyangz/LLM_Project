import unittest
import tempfile
import os
import json
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
import threading
import time
from pathlib import Path
import sys

# 添加项目根目录到Python路径，而不是src目录
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# 现在可以正确导入questions_processing模块
from src.questions_processing import QuestionsProcessor

class TestQuestionsProcessor(unittest.TestCase):
    
    def setUp(self):
        """测试前的准备工作"""
        # 创建临时目录
        self.temp_dir = tempfile.mkdtemp()
        self.vector_db_dir = os.path.join(self.temp_dir, "vector_db")
        self.documents_dir = os.path.join(self.temp_dir, "documents")
        self.questions_file = os.path.join(self.temp_dir, "questions.json")
        self.subset_file = os.path.join(self.temp_dir, "subset.csv")
        
        # 创建目录
        os.makedirs(self.vector_db_dir, exist_ok=True)
        os.makedirs(self.documents_dir, exist_ok=True)
        
        # 创建测试用的问题文件
        test_questions = [
            {"question": "腾讯的营业收入是多少？", "kind": "number"},
            {"question": "比较阿里巴巴和腾讯的净利润", "kind": "comparative"},
            {"question": "百度是否盈利？", "kind": "boolean"}
        ]
        with open(self.questions_file, 'w', encoding='utf-8') as f:
            json.dump(test_questions, f, ensure_ascii=False)
        
        # 创建测试用的subset文件
        subset_data = {
            'company_name': ['腾讯', '阿里巴巴', '百度'],
            'sha1': ['abc123', 'def456', 'ghi789']
        }
        pd.DataFrame(subset_data).to_csv(self.subset_file, index=False, encoding='utf-8')
        
        # Mock OpenAI processor
        self.mock_openai_processor = Mock()
        self.mock_openai_processor.response_data = {"usage": {"total_tokens": 100}}
    
    def tearDown(self):
        """测试后的清理工作"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_01_initialization_basic(self):
        """测试基本初始化"""
        processor = QuestionsProcessor(
            vector_db_dir=self.vector_db_dir,
            documents_dir=self.documents_dir,
            questions_file=self.questions_file
        )
        
        self.assertEqual(processor.vector_db_dir, self.vector_db_dir)
        self.assertEqual(processor.documents_dir, self.documents_dir)
        self.assertEqual(processor.questions_file, self.questions_file)
        self.assertFalse(processor.llm_reranking)
        self.assertFalse(processor.return_parent_pages)
        self.assertEqual(processor.top_n_retrieval, 10)
    
    def test_02_initialization_with_options(self):
        """测试带选项的初始化"""
        processor = QuestionsProcessor(
            vector_db_dir=self.vector_db_dir,
            documents_dir=self.documents_dir,
            questions_file=self.questions_file,
            subset_path=self.subset_file,
            llm_reranking=True,
            return_parent_pages=True,
            top_n_retrieval=20,
            new_challenge_pipeline=True
        )
        
        self.assertEqual(processor.subset_path, self.subset_file)
        self.assertTrue(processor.llm_reranking)
        self.assertTrue(processor.return_parent_pages)
        self.assertEqual(processor.top_n_retrieval, 20)
        self.assertTrue(processor.new_challenge_pipeline)
    
    def test_03_load_questions(self):
        """测试加载问题文件"""
        processor = QuestionsProcessor(
            vector_db_dir=self.vector_db_dir,
            documents_dir=self.documents_dir,
            questions_file=self.questions_file
        )
        
        questions = processor._load_questions()
        self.assertEqual(len(questions), 3)
        self.assertEqual(questions[0]["question"], "腾讯的营业收入是多少？")
        self.assertEqual(questions[1]["kind"], "comparative")
    
    def test_04_load_questions_file_not_found(self):
        """测试加载不存在的问题文件"""
        processor = QuestionsProcessor(
            vector_db_dir=self.vector_db_dir,
            documents_dir=self.documents_dir,
            questions_file="nonexistent.json"
        )
        
        with self.assertRaises(FileNotFoundError):
            processor._load_questions()
    
    def test_05_extract_companies_from_subset(self):
        """测试从subset中提取公司名"""
        processor = QuestionsProcessor(
            vector_db_dir=self.vector_db_dir,
            documents_dir=self.documents_dir,
            questions_file=self.questions_file,
            subset_path=self.subset_file,
            new_challenge_pipeline=True
        )
        
        # 测试单公司提取
        companies = processor._extract_companies_from_subset("腾讯的营业收入是多少？")
        self.assertEqual(companies, ["腾讯"])
        
        # 测试多公司提取
        companies = processor._extract_companies_from_subset("比较阿里巴巴和腾讯的净利润")
        self.assertIn("阿里巴巴", companies)
        self.assertIn("腾讯", companies)
        
        # 测试无公司
        companies = processor._extract_companies_from_subset("这是一个没有公司名的问题")
        self.assertEqual(companies, [])
    
    def test_06_extract_companies_no_subset(self):
        """测试没有subset文件时提取公司名"""
        processor = QuestionsProcessor(
            vector_db_dir=self.vector_db_dir,
            documents_dir=self.documents_dir,
            questions_file=self.questions_file,
            new_challenge_pipeline=True
        )
        
        with self.assertRaises(ValueError) as context:
            processor._extract_companies_from_subset("腾讯的营业收入是多少？")
        
        self.assertIn("subset_path must be provided", str(context.exception))
    
    def test_07_create_answer_detail_ref(self):
        """测试创建答案详情引用"""
        processor = QuestionsProcessor(
            vector_db_dir=self.vector_db_dir,
            documents_dir=self.documents_dir,
            questions_file=self.questions_file
        )
        
        detail_content = {"test": "content"}
        ref_id = processor._create_answer_detail_ref(0, detail_content)
        
        self.assertIsInstance(ref_id, str)
        self.assertIn(ref_id, processor.answer_details)
        self.assertEqual(processor.answer_details[ref_id], detail_content)
    
    def test_08_calculate_statistics(self):
        """测试统计计算"""
        processor = QuestionsProcessor(
            vector_db_dir=self.vector_db_dir,
            documents_dir=self.documents_dir,
            questions_file=self.questions_file
        )
        
        # 模拟处理结果
        results = [
            {"answer": "正常答案"},
            {"answer": None, "error": "错误信息"},
            {"answer": "N/A"},
            {"value": "正常值"},  # new_challenge_pipeline格式
            {"value": None, "error": "错误"}
        ]
        
        stats = processor._calculate_statistics(results)
        
        self.assertEqual(stats["total"], 5)
        self.assertEqual(stats["errors"], 2)
        self.assertEqual(stats["na_answers"], 1)
        self.assertEqual(stats["successful"], 2)
    
    @patch('questions_processing.VectorRetriever')
    def test_09_get_answer_for_company_basic(self, mock_retriever_class):
        """测试单公司答案获取"""
        # Mock retriever
        mock_retriever = Mock()
        mock_retriever.retrieve_by_company_name.return_value = [
            {"page": 1, "text": "测试文本1"},
            {"page": 2, "text": "测试文本2"}
        ]
        mock_retriever_class.return_value = mock_retriever
        
        # Mock OpenAI processor
        mock_answer = {
            "answer": "测试答案",
            "relevant_pages": [1, 2]
        }
        
        processor = QuestionsProcessor(
            vector_db_dir=self.vector_db_dir,
            documents_dir=self.documents_dir,
            questions_file=self.questions_file,
            subset_path=self.subset_file,
            new_challenge_pipeline=True
        )
        processor.openai_processor = self.mock_openai_processor
        processor.openai_processor.get_answer_from_rag_context.return_value = mock_answer
        
        result = processor.get_answer_for_company("腾讯", "营业收入是多少？", "number")
        
        self.assertIn("answer", result)
        self.assertIn("references", result)
        mock_retriever.retrieve_by_company_name.assert_called_once()
    
    def test_10_validate_page_references(self):
        """测试页码引用验证"""
        processor = QuestionsProcessor(
            vector_db_dir=self.vector_db_dir,
            documents_dir=self.documents_dir,
            questions_file=self.questions_file
        )
        
        retrieval_results = [
            {"page": 1, "text": "文本1"},
            {"page": 2, "text": "文本2"},
            {"page": 3, "text": "文本3"}
        ]
        
        # 测试正常情况
        validated = processor._validate_page_references([1, 2], retrieval_results)
        self.assertEqual(validated, [1, 2])
        
        # 测试幻觉页码过滤
        validated = processor._validate_page_references([1, 5, 2], retrieval_results)
        self.assertEqual(set(validated), {1, 2})
        
        # 测试最小页数补充
        validated = processor._validate_page_references([1], retrieval_results, min_pages=2)
        self.assertGreaterEqual(len(validated), 2)
    
    def test_11_extract_references(self):
        """测试引用信息提取"""
        processor = QuestionsProcessor(
            vector_db_dir=self.vector_db_dir,
            documents_dir=self.documents_dir,
            questions_file=self.questions_file,
            subset_path=self.subset_file
        )
        
        references = processor._extract_references([1, 2], "腾讯")
        
        self.assertEqual(len(references), 2)
        self.assertEqual(references[0]["pdf_sha1"], "abc123")
        self.assertEqual(references[0]["page_index"], 1)
        self.assertEqual(references[1]["page_index"], 2)
    
    def test_12_format_retrieval_results(self):
        """测试检索结果格式化"""
        processor = QuestionsProcessor(
            vector_db_dir=self.vector_db_dir,
            documents_dir=self.documents_dir,
            questions_file=self.questions_file
        )
        
        retrieval_results = [
            {"page": 1, "text": "文本1"},
            {"page": 2, "text": "文本2"}
        ]
        
        formatted = processor._format_retrieval_results(retrieval_results)
        
        self.assertIn("Text retrieved from page 1", formatted)
        self.assertIn("文本1", formatted)
        self.assertIn("---", formatted)
    
    def test_13_handle_processing_error(self):
        """测试错误处理"""
        processor = QuestionsProcessor(
            vector_db_dir=self.vector_db_dir,
            documents_dir=self.documents_dir,
            questions_file=self.questions_file,
            new_challenge_pipeline=True
        )
        
        error = ValueError("测试错误")
        result = processor._handle_processing_error(error, "测试问题", "string", 0)
        
        self.assertIn("error", result)
        self.assertIn("ValueError", result["error"])
        self.assertEqual(result["question_text"], "测试问题")
        self.assertEqual(result["kind"], "string")
        self.assertIsNone(result["value"])
    
    @patch('questions_processing.VectorRetriever')
    def test_14_process_single_question_single_company(self, mock_retriever_class):
        """测试单公司问题处理"""
        # Mock retriever
        mock_retriever = Mock()
        mock_retriever.retrieve_by_company_name.return_value = [
            {"page": 1, "text": "测试文本"}
        ]
        mock_retriever_class.return_value = mock_retriever
        
        processor = QuestionsProcessor(
            vector_db_dir=self.vector_db_dir,
            documents_dir=self.documents_dir,
            questions_file=self.questions_file,
            subset_path=self.subset_file,
            new_challenge_pipeline=True
        )
        processor.openai_processor = self.mock_openai_processor
        processor.openai_processor.get_answer_from_rag_context.return_value = {
            "answer": "测试答案",
            "relevant_pages": [1]
        }
        
        result = processor.process_single_question("腾讯的营业收入是多少？", "number")
        
        self.assertIn("answer", result)
    
    def test_15_process_single_question_no_company(self):
        """测试无公司名问题处理"""
        processor = QuestionsProcessor(
            vector_db_dir=self.vector_db_dir,
            documents_dir=self.documents_dir,
            questions_file=self.questions_file,
            subset_path=self.subset_file,
            new_challenge_pipeline=True
        )
        
        with self.assertRaises(ValueError) as context:
            processor.process_single_question("这是什么？", "string")
        
        self.assertIn("No company name found", str(context.exception))
    
    @patch('questions_processing.VectorRetriever')
    def test_16_process_comparative_question(self, mock_retriever_class):
        """测试比较问题处理"""
        # Mock retriever
        mock_retriever = Mock()
        mock_retriever.retrieve_by_company_name.return_value = [
            {"page": 1, "text": "测试文本"}
        ]
        mock_retriever_class.return_value = mock_retriever
        
        processor = QuestionsProcessor(
            vector_db_dir=self.vector_db_dir,
            documents_dir=self.documents_dir,
            questions_file=self.questions_file,
            subset_path=self.subset_file
        )
        processor.openai_processor = self.mock_openai_processor
        
        # Mock rephrased questions
        processor.openai_processor.get_rephrased_questions.return_value = {
            "腾讯": "腾讯的净利润是多少？",
            "阿里巴巴": "阿里巴巴的净利润是多少？"
        }
        
        # Mock individual answers
        processor.openai_processor.get_answer_from_rag_context.return_value = {
            "answer": "测试答案",
            "references": [{"pdf_sha1": "abc123", "page_index": 1}]
        }
        
        result = processor.process_comparative_question(
            "比较腾讯和阿里巴巴的净利润", 
            ["腾讯", "阿里巴巴"], 
            "comparative"
        )
        
        self.assertIn("references", result)
    
    def test_17_post_process_submission_answers(self):
        """测试提交答案后处理"""
        processor = QuestionsProcessor(
            vector_db_dir=self.vector_db_dir,
            documents_dir=self.documents_dir,
            questions_file=self.questions_file
        )
        
        answers = [
            {
                "value": "测试答案",
                "references": [{"pdf_sha1": "abc123", "page_index": 0}]  # 0-based
            },
            {
                "value": "N/A",
                "references": [{"pdf_sha1": "def456", "page_index": 1}]
            }
        ]
        
        processed = processor._post_process_submission_answers(answers)
        
        # 检查页码转换（0-based -> 1-based）
        self.assertEqual(processed[0]["references"][0]["page_index"], 1)
        
        # 检查N/A答案的引用清空
        self.assertEqual(processed[1]["references"], [])
        
        # 检查step_by_step_analysis字段添加
        self.assertIn("step_by_step_analysis", processed[0])
    
    def test_18_save_progress(self):
        """测试进度保存"""
        processor = QuestionsProcessor(
            vector_db_dir=self.vector_db_dir,
            documents_dir=self.documents_dir,
            questions_file=self.questions_file
        )
        
        results = [{"answer": "测试答案"}]
        progress_file = os.path.join(self.temp_dir, "progress.json")
        
        processor._save_progress(results, progress_file)
        
        # 检查文件是否创建
        self.assertTrue(os.path.exists(progress_file))
        
        # 检查内容
        with open(progress_file, 'r', encoding='utf-8') as f:
            saved_data = json.load(f)
        
        self.assertIn("results", saved_data)
        self.assertIn("statistics", saved_data)
        self.assertIn("answer_details", saved_data)
    
    def test_19_thread_safety(self):
        """测试线程安全性"""
        processor = QuestionsProcessor(
            vector_db_dir=self.vector_db_dir,
            documents_dir=self.documents_dir,
            questions_file=self.questions_file
        )
        
        def create_ref(index):
            return processor._create_answer_detail_ref(index, {"data": f"test_{index}"})
        
        # 并发创建引用
        threads = []
        results = []
        
        for i in range(10):
            thread = threading.Thread(target=lambda i=i: results.append(create_ref(i)))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # 检查所有引用都被正确创建
        self.assertEqual(len(results), 10)
        self.assertEqual(len(set(results)), 10)  # 所有引用ID都不同
    
    def test_20_encoding_handling(self):
        """测试编码处理"""
        # 创建GBK编码的subset文件
        gbk_subset_file = os.path.join(self.temp_dir, "subset_gbk.csv")
        subset_data = {
            'company_name': ['腾讯控股', '阿里巴巴集团'],
            'sha1': ['abc123', 'def456']
        }
        pd.DataFrame(subset_data).to_csv(gbk_subset_file, index=False, encoding='gbk')
        
        processor = QuestionsProcessor(
            vector_db_dir=self.vector_db_dir,
            documents_dir=self.documents_dir,
            questions_file=self.questions_file,
            subset_path=gbk_subset_file,
            new_challenge_pipeline=True
        )
        
        # 应该能正确处理GBK编码
        companies = processor._extract_companies_from_subset("腾讯控股的营业收入是多少？")
        self.assertEqual(companies, ["腾讯控股"])

def run_all_tests():
    """运行所有测试"""
    unittest.main(verbosity=2)

if __name__ == "__main__":
    run_all_tests()