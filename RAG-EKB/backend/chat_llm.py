from openai import OpenAI
import os

# 初始化 OpenAI 客户端
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 定义一个函数，用于生成代码
def generate_code(prompt):
    # 使用 OpenAI 的 Completion API 生成代码
    response = client.chat.completions.create(
        model="qwen-turbo",
        messages=[
            {"role": "system", "content": "你是一个代码助手，请帮助生成代码。"},
            {"role": "user", "content": prompt}
        ],
        max_tokens=100,
        temperature=0.5,
    )
    return response.choices[0].message.content

print(generate_code("请用python写一个hello world"))

