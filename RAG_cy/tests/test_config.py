#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试配置文件

定义测试相关的配置参数和常量
"""

from pathlib import Path

# 测试目录配置
TEST_ROOT = Path(__file__).parent
PROJECT_ROOT = TEST_ROOT.parent
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"
DEBUG_DATA_DIR = PROJECT_ROOT / "debug_data"

# PDF测试文件配置
PDF_TEST_DIR = DATA_DIR / "pdf_reports"
TEST_OUTPUT_DIR = DEBUG_DATA_DIR / "test_output"
TEST_CSV_FILE = DEBUG_DATA_DIR / "test_metadata.csv"

# 测试参数配置
MAX_TEST_FILES = 5  # 最大测试文件数量
TEST_TIMEOUT = 300  # 测试超时时间（秒）
PERF_TEST_FILES = 3  # 性能测试文件数量

# 日志配置
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = TEST_ROOT / "test.log"

# 测试数据配置
TEST_CSV_CONTENT = '''sha1,company_name,file_name
abc123def456,"测试公司A",test_report_a.pdf
789xyz012uvw,"测试公司B",test_report_b.pdf
test_hash_001,"中芯国际",中芯国际机构调研纪要.pdf
test_hash_002,"上海证券",上海证券研究报告.pdf
test_hash_003,"东方证券",东方证券研究报告.pdf
test_hash_004,"华泰证券",华泰证券研究报告.pdf
'''

# 预期的输出JSON结构
EXPECTED_JSON_KEYS = ['metainfo', 'content', 'tables', 'pictures']
EXPECTED_METAINFO_KEYS = ['sha1']
OPTIONAL_METAINFO_KEYS = ['company_name']