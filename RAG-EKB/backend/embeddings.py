from fastapi import FastAPI
import faiss 
import  numpy as np


#初始化Faiss索引
d = 512 #向量维度
index = faiss.IndexFlatL2(d)

#添加向量到索引
x = np.random.random((100, d)).astype('float32')
index.add(x)

#查询向量
query_embedding = np.random.random((1, d)).astype('float32')
k = 5 #返回前k个最相似的向量
D, I = index.search(query_embedding, k)

#打印结果
print("最相似的向量的索引:", I)
print("最相似的向量的距离:", D)



