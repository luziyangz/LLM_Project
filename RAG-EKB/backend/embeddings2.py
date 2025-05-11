import faiss
import PyPDF2
from pathlib import Path  # 添加这行导入
import torch  # 需要添加，因为后面使用了PyTorch
import os

#定义一个函数，读取目标目录下的所有类型的文件（支持pdf\word\txt等文本类型），并返回一个列表，列表中每个元素是一个字典，包含文件名和文件内容
#支持的文件类型包括PDF、TXT、DOC/DOCX等常见文本格式
#返回的列表中每个元素为字典，包含文件名(filename)和文件内容(content)两个键值对
def read_files(target_dir):
    files = []
    for file in target_dir.iterdir():
        if file.is_file():
            if file.suffix.lower() == '.pdf':
                try:
                    # 使用PyPDF2读取PDF文件
                    pdf_reader = PyPDF2.PdfReader(str(file))
                    content = ""
                    # 遍历所有页面并提取文本
                    for page in pdf_reader.pages:
                        content += page.extract_text()
                    files.append({"filename": file.name, "content": content})
                except Exception as e:
                    print(f"警告：无法读取PDF文件 {file.name}，错误：{str(e)}")
                    continue
            else:
                try:
                    # 处理其他文本文件
                    with open(file, "r", encoding="utf-8") as f:
                        content = f.read()
                except UnicodeDecodeError:
                    try:
                        with open(file, "r", encoding="gbk") as f:
                            content = f.read()
                    except UnicodeDecodeError:
                        print(f"警告：无法读取文件 {file.name}，编码不支持")
                        continue
                files.append({"filename": file.name, "content": content})
    return files

#定义一个函数，将文件内容分割成指定大小的块以及重复块，并返回一个列表，列表中每个元素是一个字典，包含文件名、文件内容和块的编号
def split_files(files, chunk_size=1000, overlap=200):
    chunks = []
    for file in files:
        content = file["content"]
        filename = file["filename"]
        
        # 计算实际的步进大小（考虑重叠）
        stride = chunk_size - overlap
        
        # 使用滑动窗口进行分块
        for i in range(0, len(content), stride):
            chunk = content[i:i+chunk_size]
            
            # 确保最后一个块不会太小
            if len(chunk) < chunk_size/2 and i > 0:
                break
                
            chunks.append({
                "filename": filename,
                "content": chunk,
                "chunk_id": i//stride,
                "start_char": i,
                "end_char": i+len(chunk)
            })
    return chunks

#定义一个函数，选择嵌入模型，将文本块嵌入到向量数据库中，并返回一个向量数据库对象，支持本地存储，位置在：E:/code/AIProjectCode/trae_code/project/RAG-EKB/backend/data/vector_store/faiss_index
def embed_chunks(chunks, embedding_model):
    # 选择嵌入模型
    if embedding_model == "bg3":
        from openai import OpenAI
        import os
        import numpy as np
        
        # 添加SSL验证禁用（仅用于测试，生产环境建议properly配置SSL证书）
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context
        
        client = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        
        def embed_text(text):
            try:
                response = client.embeddings.create(
                    model="text-embedding-v1",
                    input=text,
                    timeout=30  # 添加超时设置
                )
                return response.data[0].embedding
            except Exception as e:
                print(f"嵌入处理错误：{str(e)}")
                raise
            
    else:
        raise ValueError("不支持的嵌入模型")

    # 将文本块嵌入到向量数据库中
    embeddings = []
    for chunk in chunks:
        try:
            embedding = embed_text(chunk["content"])
            embeddings.append(embedding)
        except Exception as e:
            print(f"处理文本块时出错：{str(e)}")
            continue
    
    if not embeddings:
        raise ValueError("没有成功生成任何嵌入向量")
    
    # 确保embeddings是numpy数组并且类型是float32
    embeddings = np.array(embeddings).astype('float32')
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    
    # 创建存储目录（如果不存在）
    save_dir = Path("E:/code/AIProjectCode/trae_code/project/RAG-EKB/backend/data/vector_store/faiss_index")
    save_dir.parent.mkdir(parents=True, exist_ok=True)
    
    # 保存索引到磁盘
    faiss.write_index(index, str(save_dir))
    print(f"向量索引已保存到：{save_dir}")
    
    return index

#输出结果
target_dir = Path("E:/code/AIProjectCode/trae_code/project/RAG-EKB/backend/file")
files = read_files(target_dir)
chunks = split_files(files)
index = embed_chunks(chunks, "bg3")
print(index)

    





