import requests
import os
import time
import zipfile

api_key = 'eyJ0eXBlIjoiSldUIiwiYWxnIjoiSFM1MTIifQ.eyJqdGkiOiI4NTUwMzQ1NSIsInJvbCI6IlJPTEVfUkVHSVNURVIiLCJpc3MiOiJPcGVuWExhYiIsImlhdCI6MTc1MTI4ODI0OCwiY2xpZW50SWQiOiJsa3pkeDU3bnZ5MjJqa3BxOXgydyIsInBob25lIjoiMTc2MjQyMTgzMTciLCJvcGVuSWQiOm51bGwsInV1aWQiOiJjODM4ZGRlMS1kYjc1LTQxMWQtODE0OS0xMDE1MzJmMzI5MzYiLCJlbWFpbCI6IiIsImV4cCI6MTc1MjQ5Nzg0OH0.I5Q_7bhOIPen8GsaAsh1zaV89wnHfOCFLR9PNvbI3tEVt-NPHl6ZvB3iOcSOG0eZthf3Kyq_ezRbQ0RROouebQ'
def get_task_id(file_name):
    url='https://mineru.net/api/v4/extract/task'
    header = {
        'Content-Type':'application/json',
        "Authorization":f"Bearer {api_key}"
    }
    pdf_url = 'https://vl-image.oss-cn-shanghai.aliyuncs.com/pdf/' + file_name
    data = {
        'url':pdf_url,
        'is_ocr':True,
        'enable_formula': False,
    }

    try:
        res = requests.post(url, headers=header, json=data)
        print(f"状态码: {res.status_code}")
        
        # 检查HTTP状态码
        if res.status_code != 200:
            print(f"API请求失败，状态码: {res.status_code}")
            print(f"响应内容: {res.text}")
            if res.status_code == 401:
                print("认证失败，请检查API key是否有效")
            return None
        
        # 解析响应
        response_data = res.json()
        print(f"响应数据: {response_data}")
        
        # 修复：检查code字段而不是success字段
        if response_data.get('code') != 0:
            print(f"API调用失败: {response_data.get('msg', '未知错误')}")
            return None
            
        if 'data' not in response_data or not response_data['data']:
            print("响应中缺少data字段或data为空")
            return None
            
        if 'task_id' not in response_data['data']:
            print("响应中缺少task_id字段")
            return None
        
        task_id = response_data['data']['task_id']
        print(f"获取到task_id: {task_id}")
        return task_id
        
    except requests.exceptions.RequestException as e:
        print(f"网络请求异常: {e}")
        return None
    except ValueError as e:
        print(f"JSON解析失败: {e}")
        return None
    except Exception as e:
        print(f"未知错误: {e}")
        return None

def get_result(task_id):
    if not task_id:
        print("task_id为空，无法获取结果")
        return None
        
    url = f'https://mineru.net/api/v4/extract/task/{task_id}'
    header = {
        'Content-Type':'application/json',
        "Authorization":f"Bearer {api_key}"
    }

    while True:
        try:
            res = requests.get(url, headers=header)
            
            if res.status_code != 200:
                print(f"获取结果失败，状态码: {res.status_code}")
                return None
                
            response_data = res.json()
            
            if 'data' not in response_data:
                print("响应中缺少data字段")
                return None
                
            result = response_data["data"]
            print(result)
            
            state = result.get('state')
            err_msg = result.get('err_msg', '')
            
            # 如果任务还在进行中，等待后重试
            if state in ['pending', 'running']:
                print("任务未完成，等待5秒后重试...")
                time.sleep(5)
                continue
                
            # 如果有错误，输出错误信息
            if err_msg:
                print(f"任务出错: {err_msg}")
                return None
                
            # 如果任务完成，下载并解压结果
            if state == 'done':
                # 修复：使用正确的字段名 'full_zip_url' 而不是 'download_url'
                download_url = result.get('full_zip_url')
                if download_url:
                    # 下载zip文件
                    zip_filename = f"{task_id}.zip"
                    download_response = requests.get(download_url)
                    with open(zip_filename, 'wb') as f:
                        f.write(download_response.content)
                    print(f"下载完成: {zip_filename}")
                    
                    # 解压文件
                    unzip_file(zip_filename, task_id)
                    return result
                else:
                    print("未找到下载链接")
                    return None
                
        except Exception as e:
            print(f"获取结果时发生错误: {e}")
            return None

# 解压zip文件的函数
def unzip_file(zip_path, extract_dir=None):
    """
    解压指定的zip文件到目标文件夹。
    :param zip_path: zip文件路径
    :param extract_dir: 解压目标文件夹，默认为zip同名目录
    """
    if extract_dir is None:
        extract_dir = zip_path.rstrip('.zip')
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    print(f"已解压到: {extract_dir}")

if __name__ == "__main__":
    file_name = '【财报】中芯国际：中芯国际2024年年度报告.pdf'
    task_id = get_task_id(file_name)
    print('task_id:',task_id)
    get_result(task_id)
