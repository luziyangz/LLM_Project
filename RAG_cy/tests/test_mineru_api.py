#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mineru API 简单测试程序
用于测试 Mineru API 的连接性和基本功能
"""

import sys
import os

# 添加src目录到路径，以便导入pdf_mineru模块
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import requests
import json

# 从pdf_mineru模块导入API key
try:
    from pdf_mineru import api_key
except ImportError:
    print("错误：无法导入pdf_mineru模块，请确保src目录中存在pdf_mineru.py文件")
    sys.exit(1)

def test_api_connection():
    """
    测试API连接性 - 简单的健康检查
    """
    print("=== 测试API连接性 ===")
    
    # 测试用的简单URL（如果有健康检查端点的话）
    test_url = 'https://mineru.net/api/v4/extract/task'
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }
    
    try:
        # 发送一个简单的POST请求测试认证
        test_data = {
            'url': 'https://example.com/test.pdf',  # 测试用的假URL
            'is_ocr': False,
            'enable_formula': False,
        }
        
        response = requests.post(test_url, headers=headers, json=test_data, timeout=10)
        
        print(f"状态码: {response.status_code}")
        print(f"响应头: {dict(response.headers)}")
        
        if response.status_code == 200:
            print("✅ API连接成功！")
            try:
                data = response.json()
                print(f"响应数据: {json.dumps(data, indent=2, ensure_ascii=False)}")
                return True
            except json.JSONDecodeError:
                print("⚠️  响应不是有效的JSON格式")
                print(f"响应内容: {response.text}")
                return False
        elif response.status_code == 401:
            print("❌ 认证失败！请检查API key是否有效")
            print(f"响应内容: {response.text}")
            return False
        elif response.status_code == 400:
            print("⚠️  请求参数错误（这是预期的，因为我们使用了测试URL）")
            print(f"响应内容: {response.text}")
            return True  # 400错误说明认证通过了，只是参数有问题
        else:
            print(f"❌ API请求失败，状态码: {response.status_code}")
            print(f"响应内容: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("❌ 请求超时，请检查网络连接")
        return False
    except requests.exceptions.ConnectionError:
        print("❌ 连接错误，请检查网络连接")
        return False
    except Exception as e:
        print(f"❌ 未知错误: {e}")
        return False

def test_api_key_format():
    """
    测试API key格式
    """
    print("\n=== 测试API Key格式 ===")
    
    if not api_key:
        print("❌ API key为空")
        return False
    
    if not api_key.startswith('eyJ'):
        print("❌ API key格式不正确（应该是JWT格式，以eyJ开头）")
        return False
    
    if len(api_key) < 100:
        print("❌ API key长度太短，可能不完整")
        return False
    
    print(f"✅ API key格式看起来正确")
    print(f"API key长度: {len(api_key)}")
    print(f"API key前缀: {api_key[:20]}...")
    return True

def test_requests_module():
    """
    测试requests模块是否可用
    """
    print("\n=== 测试依赖模块 ===")
    
    try:
        import requests
        print(f"✅ requests模块版本: {requests.__version__}")
        return True
    except ImportError:
        print("❌ requests模块未安装，请运行: pip install requests")
        return False

def main():
    """
    主测试函数
    """
    print("Mineru API 测试程序")
    print("=" * 50)
    
    # 测试结果
    results = []
    
    # 1. 测试依赖模块
    results.append(test_requests_module())
    
    # 2. 测试API key格式
    results.append(test_api_key_format())
    
    # 3. 测试API连接
    results.append(test_api_connection())
    
    # 总结
    print("\n" + "=" * 50)
    print("测试总结:")
    
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"✅ 所有测试通过 ({passed}/{total})")
        print("Mineru API 可以正常使用！")
    else:
        print(f"❌ 部分测试失败 ({passed}/{total})")
        print("请根据上述错误信息修复问题后重试")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)