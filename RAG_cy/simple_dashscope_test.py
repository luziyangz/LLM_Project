import os
import time
import json
from dotenv import load_dotenv
from openai import OpenAI
import logging

# 加载环境变量
load_dotenv()

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DashScopeSimpleTester:
    """阿里云DashScope简单测试器"""
    
    def __init__(self):
        self.api_key = os.getenv("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError("请在.env文件中设置DASHSCOPE_API_KEY")
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
    
    def test_chat_model(self, model_name="qwen-turbo", test_prompt="你好，请简单介绍一下你自己。"):
        """测试聊天模型"""
        print(f"🔍 测试聊天模型: {model_name}")
        
        result = {
            "model": model_name,
            "type": "chat",
            "status": "failed",
            "response_time": 0,
            "error": None,
            "response": None,
            "usage": None
        }
        
        try:
            start_time = time.time()
            
            response = self.client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "user", "content": test_prompt}
                ],
                max_tokens=100,
                temperature=0.7
            )
            
            end_time = time.time()
            result["response_time"] = round(end_time - start_time, 2)
            result["status"] = "success"
            result["response"] = response.choices[0].message.content
            result["usage"] = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
            
            print(f"✅ 测试成功！响应时间: {result['response_time']}秒")
            print(f"📝 响应内容: {result['response'][:100]}...")
            print(f"📊 Token使用: {result['usage']}")
            
        except Exception as e:
            result["error"] = str(e)
            print(f"❌ 测试失败: {e}")
            logger.error(f"聊天模型 {model_name} 测试失败: {e}")
        
        return result
    
    def test_embedding_model(self, model_name="text-embedding-v1", test_text="这是一个测试文本"):
        """测试嵌入模型"""
        print(f"\n🔍 测试嵌入模型: {model_name}")
        
        result = {
            "model": model_name,
            "type": "embedding",
            "status": "failed",
            "response_time": 0,
            "error": None,
            "embedding_dim": None
        }
        
        try:
            start_time = time.time()
            
            response = self.client.embeddings.create(
                model=model_name,
                input=test_text
            )
            
            end_time = time.time()
            result["response_time"] = round(end_time - start_time, 2)
            result["status"] = "success"
            result["embedding_dim"] = len(response.data[0].embedding)
            
            print(f"✅ 嵌入模型测试成功！响应时间: {result['response_time']}秒")
            print(f"📊 嵌入维度: {result['embedding_dim']}")
            
        except Exception as e:
            result["error"] = str(e)
            print(f"❌ 嵌入模型测试失败: {e}")
            logger.error(f"嵌入模型 {model_name} 测试失败: {e}")
        
        return result
    
    def run_all_tests(self):
        """运行所有测试"""
        print("🚀 阿里云DashScope API测试")
        print("=" * 40)
        
        results = []
        
        # 测试可用的聊天模型
        chat_models = [
            "qwen-turbo",
            "qwen-plus",
            "qwen-max",
            "qwen-long"
        ]
        
        for model in chat_models:
            result = self.test_chat_model(model)
            results.append(result)
            time.sleep(1)  # 避免请求过于频繁
        
        # 测试嵌入模型
        embedding_models = [
            "text-embedding-v1",
            "text-embedding-v2"
        ]
        
        for model in embedding_models:
            result = self.test_embedding_model(model)
            results.append(result)
            time.sleep(1)
        
        return results
    
    def generate_report(self, results):
        """生成测试报告"""
        report = []
        report.append("=" * 50)
        report.append("阿里云DashScope API测试报告")
        report.append("=" * 50)
        report.append(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"测试模型数量: {len(results)}")
        report.append("")
        
        success_count = sum(1 for r in results if r["status"] == "success")
        report.append(f"成功: {success_count}/{len(results)}")
        report.append("")
        
        for result in results:
            report.append("-" * 30)
            report.append(f"模型: {result['model']}")
            report.append(f"类型: {result['type']}")
            report.append(f"状态: {result['status']}")
            report.append(f"响应时间: {result['response_time']}秒")
            
            if result["status"] == "success":
                if "response" in result and result["response"]:
                    report.append(f"响应内容: {result['response'][:80]}...")
                if "usage" in result and result["usage"]:
                    report.append(f"Token使用: {result['usage']}")
                if "embedding_dim" in result and result["embedding_dim"]:
                    report.append(f"嵌入维度: {result['embedding_dim']}")
            else:
                report.append(f"错误信息: {result['error']}")
            
            report.append("")
        
        return "\n".join(report)
    
    def save_results(self, results, filename="dashscope_test_results.json"):
        """保存测试结果"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump({
                    "test_time": time.strftime('%Y-%m-%d %H:%M:%S'),
                    "results": results
                }, f, ensure_ascii=False, indent=2)
            print(f"\n💾 测试结果已保存到: {filename}")
        except Exception as e:
            print(f"❌ 保存结果失败: {e}")

def quick_test():
    """快速测试函数"""
    print("⚡ 快速测试阿里云DashScope API")
    print("=" * 35)
    
    try:
        tester = DashScopeSimpleTester()
        
        # 快速测试一个聊天模型
        chat_result = tester.test_chat_model("qwen-turbo", "你好")
        
        # 快速测试嵌入模型
        embedding_result = tester.test_embedding_model("text-embedding-v1", "测试文本")
        
        success_count = sum(1 for r in [chat_result, embedding_result] if r["status"] == "success")
        print(f"\n📊 快速测试结果: {success_count}/2 个模型连接成功")
        
        return success_count == 2
        
    except Exception as e:
        print(f"❌ 快速测试失败: {e}")
        return False

def main():
    """主函数"""
    try:
        tester = DashScopeSimpleTester()
        
        # 运行所有测试
        results = tester.run_all_tests()
        
        # 生成报告
        report = tester.generate_report(results)
        print("\n" + report)
        
        # 保存结果
        tester.save_results(results)
        
        # 保存报告
        with open("dashscope_test_report.txt", 'w', encoding='utf-8') as f:
            f.write(report)
        
        print("\n✅ 测试完成！")
        
    except Exception as e:
        print(f"❌ 程序运行失败: {e}")
        logger.error(f"程序运行失败: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        quick_test()
    else:
        main()