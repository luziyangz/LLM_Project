#使用fastapi 创建一个后端服务器
from fastapi import FastAPI


app = FastAPI()

#定义一个接口，用于上传文件，支持txt、pdf\word等文件类型
