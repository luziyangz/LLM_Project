from fastapi import FastAPI, Body
from pydantic import BaseModel
import os
import json
import subprocess
import logging  # 确保logging在最前面导入
from typing import List, Optional, AsyncGenerator, Dict, Any, Union
from pydantic import validator
from datetime import datetime
import uuid # 新增导入
import time # 新增导入
import traceback # 新增导入
import asyncio
from functools import wraps
import sqlite3
import sqlalchemy.exc
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from fastapi.responses import FileResponse
from fastapi.responses import StreamingResponse
from fastapi.responses import JSONResponse
from fastapi import HTTPException
from fastapi import FastAPI, Request, Response, File, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.exceptions import RequestValidationError
from fastapi.encoders import jsonable_encoder
from utils.logs_utils import LoggerConfig, log_decorator
from utils.embedding_utils import EmbeddingModelLoader
# 在文件开头添加
from utils.db_utils import DatabaseManager

from utils.document_processor import DocumentProcessor
from utils.rag_qa_system import RAGQASystem
# 添加缺失的导入
from utils.configurable_processor import ConfigurableDocumentProcessor

# 添加全局变量
AsyncSessionLocal = None
db_manager = None
# 添加全局RAG系统实例
rag_system = None

# 定义上传目录常量
UPLOAD_DIR = "E:/code/AIProjectCode/trae_code/project/RAG-EKB/backend/data/uploads"

# 添加缺失的初始化函数
async def initdb():
    """初始化数据库"""
    global AsyncSessionLocal, db_manager
    try:
        # 定义数据库连接字符串（SQLite数据库）
        database_url = "sqlite+aiosqlite:///./chat_database.db"
        
        # 传递连接字符串给DatabaseManager
        db_manager = DatabaseManager(database_url)
        engine, session_factory = db_manager.init_db()
        AsyncSessionLocal = session_factory
        
        # 创建数据库表
        db_manager.check_and_create_tables(Base)
        
        logger.info("数据库初始化成功")
    except Exception as e:
        logger.error(f"数据库初始化失败: {e}")
        raise


from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from sqlalchemy import select # 新增导入
from sqlalchemy.orm import selectinload # 新增导入
from dao.models import Base, ChatSession, ChatMessage, Document # 将此导入移至顶部
import dashscope # 新增导入 for LLM
import asyncio # 确保 asyncio 已导入，如果其他地方需要

# 创建日志实例
logger = LoggerConfig(tag='APP-backend').get_logger()

def db_retry(max_retries=3, delay=1):
    """数据库操作重试装饰器"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except (sqlite3.OperationalError, sqlalchemy.exc.OperationalError) as e:
                    if "no active connection" in str(e) and attempt < max_retries - 1:
                        logger.warning(f"数据库连接错误，重试 {attempt + 1}/{max_retries}: {e}")
                        await asyncio.sleep(delay * (attempt + 1))
                        continue
                    raise
            return None
        return wrapper
    return decorator

# 创建应用启动上下文管理器
@asynccontextmanager
async def lifespan(app):
    # 启动前执行，初始化模型
    await init()
    await initdb()
    await load_documents()
    logger.info("服务启动成功")
    yield
    # 关闭时的清理代码可以放在这里
    logger.info("服务关闭")

# 创建一个后端服务
app = FastAPI(lifespan=lifespan)
# 挂载一个静态文件目录
app.mount("/static", StaticFiles(directory="E:/code/AIProjectCode/trae_code/project/RAG-EKB/frontend/ui"), name="static")
doc_processor = DocumentProcessor()

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    # 使用 str() 而不是 decode()
    body_str = str(exc.body) if exc.body else "No body"
    logger.error(f"请求验证失败: {exc.errors()}, 请求体: {body_str}")
    return JSONResponse(
        status_code=422,
        content={
            "detail": exc.errors(),
            "body": body_str,
            "message": "请求数据验证失败，请检查数据格式"
        }
    )

# 添加CORS中间件允许跨域请求
# 取消根路由重定向的注释
@app.get("/")
async def root():
    logger.info("根路由重定向到聊天界面")   
    return RedirectResponse(url="/static/chat.html")

# 修改CORS配置，同时允许localhost和127.0.0.1
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://127.0.0.1:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@log_decorator(level=logging.INFO)
async def init():  # 修改为异步函数
    global rag_system
    logger.info("开始初始化服务...")
    try:
        # 获取环境变量
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            logger.warning("未找到DASHSCOPE_API_KEY环境变量，请确保已正确设置")
            
        # 配置模型加载参数
        model_config = {
            "type": "huggingface",  # 首次使用huggingface下载，之后会自动变为local模式
            "path": "E:/code/AIProjectCode/trae_code/project/RAG-EKB/backend/utils/models",  # 本地模型存储路径
            "huggingface_model": "sentence-transformers/all-MiniLM-L6-v2",  # huggingface模型名称
            "cache_dir": "E:/code/AIProjectCode/trae_code/project/RAG-EKB/backend/utils/models",  # 下载缓存目录
            "api_key": api_key,  # 从环境变量获取API密钥
            "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",  # API基础URL
            "model_name": "text-embedding-v3",  # 模型名称
            "provider": "aliyun",  # 模型提供商
        }
        
        # 加载模型
        model_loader = EmbeddingModelLoader(model_config)
        model = model_loader.load_model()
        logger.info("模型加载完成")
        
        # 构建embedding配置
        embedding_config = {
            "type": "api",  # 从 "local" 改为 "api"
            "provider": "aliyun",
            "api_key": os.getenv("DASHSCOPE_API_KEY"),
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "model_name": "text-embedding-v1"
        }
        
        # 初始化RAG系统
        logger.info("正在初始化RAG系统...")
        rag_system = RAGQASystem(
            embedding_config=embedding_config,
            index_dimension=1536,  # all-MiniLM-L6-v2的维度
            chunker_type='fixed_length',
            chunker_config={},
            enable_rerank=True,  # 启用重排
            rerank_strategy='hybrid'  # 使用混合重排策略
        )
        
        # 验证初始化结果
        if rag_system is None:
            raise Exception("RAG系统初始化后仍为None")
            
        # 测试RAG系统功能
        test_stats = rag_system.get_stats()
        logger.info(f"RAG系统初始化完成，初始统计: {test_stats}")
        
        logger.info("所有服务初始化完成")
        
    except Exception as e:
        logger.error(f"初始化失败: {str(e)}", exc_info=True)
        # 确保rag_system不会保持在不完整状态
        rag_system = None
        raise


from dotenv import load_dotenv

# 加载环境变量
load_dotenv()



#创建一个加载文件函数
@log_decorator(level=logging.INFO)
async def load_documents():
    """加载文档"""
    logger.info("开始加载文档...")
    try:
        # 检查是否存在已保存的索引
        index_path = "./data/faiss_index/index.faiss"
        logger.info(f"检查索引文件: {index_path}")
        
        if os.path.exists(index_path):
            logger.info("发现已有索引，正在加载...")
            rag_system.load_index(index_path)
            stats = rag_system.get_stats()
            logger.info(f"索引加载完成 - 文档数: {stats['total_documents']}, 分块数: {stats['total_chunks']}, 索引大小: {stats['index_size']}")
        else:
            logger.warning(f"未找到索引文件: {index_path}")
            logger.info("尝试加载默认文档目录...")
            await load_default_documents()
        
        # 最终状态检查
        final_stats = rag_system.get_stats()
        if final_stats['total_chunks'] == 0:
            logger.warning("警告: 知识库为空，RAG功能将无法正常工作")
            logger.info("请通过 /api/rag/add_document 接口添加文档")
        else:
            logger.info(f"知识库加载完成 - 包含 {final_stats['total_documents']} 个文档，{final_stats['total_chunks']} 个分块")
        
    except Exception as e:
        logger.error(f"文档加载失败: {str(e)}", exc_info=True)
        raise

async def load_default_documents():
    """加载默认文档目录中的所有文档"""
    default_docs_dir = "./data/documents"
    logger.info(f"检查默认文档目录: {default_docs_dir}")
    
    if not os.path.exists(default_docs_dir):
        logger.warning(f"默认文档目录不存在: {default_docs_dir}")
        logger.info("创建默认文档目录...")
        os.makedirs(default_docs_dir, exist_ok=True)
        logger.info("请将文档文件放入该目录并重启服务")
        return
    
    files = [f for f in os.listdir(default_docs_dir) if f.endswith(('.txt', '.pdf', '.docx', '.md'))]
    
    if not files:
        logger.warning(f"默认文档目录为空: {default_docs_dir}")
        return
    
    logger.info(f"发现 {len(files)} 个文档文件，开始加载...")
    
    loaded_count = 0
    for filename in files:
        file_path = os.path.join(default_docs_dir, filename)
        try:
            logger.info(f"正在加载文档: {filename}")
            result = await rag_system.add_document(file_path, filename)
            if result.get('success', False):
                loaded_count += 1
                logger.info(f"文档加载成功: {filename}")
            else:
                logger.error(f"文档加载失败: {filename}")
        except Exception as e:
            logger.error(f"加载文档 {filename} 时出错: {e}")
    
    logger.info(f"默认文档加载完成，成功加载 {loaded_count}/{len(files)} 个文档")

# #处理前端界面的请求：当前端发送用户消息时
# @app.get("/static/")
# async def read_root():
#     return {"Hello": "World"}

#处理前端的请求
@app.get("/items/{item_id}")
async def read_item(item_id: int):
    return {"item_id": item_id}

# # 处理聊天请求的API端点
# @app.post("/api/chat")
# async def chat(message: str):
#     try:
#         # 临时返回一个简单的响应
#         return {"response": "您好，我是RAG-EKB知识库助手，很高兴为您服务！"}
#     except Exception as e:
#         logger.error(f"处理聊天请求失败: {str(e)}", exc_info=True)
#         return {"error": "服务器处理请求失败，请稍后重试"}

# # 根路由重定向到聊天界面
# @app.get("/")
# async def root():
#     logger.info("根路由重定向到聊天界面")   
#     return RedirectResponse(url="/static/chat.html")

# 健康检查接口
@app.get("/api/health")
async def health_check():
    """系统健康检查"""
    try:
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "services": {
                "rag_system": rag_system is not None,
                "database": AsyncSessionLocal is not None,
            }
        }
        
        if rag_system:
            try:
                rag_stats = rag_system.get_stats()
                health_status["rag_details"] = {
                    "initialized": True,
                    "has_index": rag_stats.get('has_index', False),
                    "total_chunks": rag_stats.get('total_chunks', 0)
                }
            except Exception as e:
                health_status["rag_details"] = {
                    "initialized": True,
                    "error": str(e)
                }
        else:
            health_status["rag_details"] = {
                "initialized": False,
                "error": "RAG系统未初始化"
            }
        
        # 如果关键服务未运行，返回unhealthy状态
        if not health_status["services"]["rag_system"]:
            health_status["status"] = "unhealthy"
            
        return health_status
        
    except Exception as e:
        logger.error(f"健康检查失败: {e}")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

# 健康检查重定向路由
@app.get("/health")
async def health_redirect():
    return RedirectResponse(url="/api/health")

# @app.post("/api/chat")
# async def chat(request: ChatRequest):
#     try:
#         return {"response": f"您好，我收到了您的消息：{request.message}"}
#     except Exception as e:
#         logger.error(f"处理聊天请求失败: {str(e)}", exc_info=True)
#         return {"error": "服务器处理请求失败，请稍后重试"}

@app.get("/api/chat/history")
async def get_chat_history(session_id: Optional[str] = None, limit: int = 50):
    """获取聊天历史记录"""
    try:
        async with db_manager.get_session() as session:
            if session_id:
                # 获取特定会话的历史记录
                result = await session.execute(
                    select(ChatMessage)
                    .where(ChatMessage.session_id == session_id)
                    .order_by(ChatMessage.created_at.desc())
                    .limit(limit)
                )
                messages = result.scalars().all()
                
                return {
                    "session_id": session_id,
                    "messages": [
                        {
                            "id": msg.id,
                            "role": msg.role,
                            "content": msg.content,
                            "created_at": msg.created_at.isoformat()
                        }
                        for msg in reversed(messages)  # 按时间正序返回
                    ]
                }
            else:
                # 获取所有会话列表
                result = await session.execute(
                    select(ChatSession)
                    .order_by(ChatSession.created_at.desc())
                    .limit(limit)
                )
                sessions = result.scalars().all()
                
                return {
                    "sessions": [
                        {
                            "id": sess.id,
                            "user_id": sess.user_id,
                            "created_at": sess.created_at.isoformat(),
                            "updated_at": sess.updated_at.isoformat() if sess.updated_at else None
                        }
                        for sess in sessions
                    ]
                }
                
    except Exception as e:
        logger.error(f"获取聊天历史记录失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="获取聊天历史记录失败")

# 处理流式SSE响应   
# 修改现有的stream_response端点，添加RAG参数
@app.get("/api/stream")
async def stream_response(
    query: str,
    session_id: Optional[str] = None,
    web_search: bool = False,
    use_rag: bool = True  # 新增RAG参数
):
    try:
        logger.info(f"收到流式请求 - 查询: {query}, 会话ID: {session_id}, 联网搜索: {web_search}, RAG: {use_rag}")

        if not query:
            raise HTTPException(status_code=400, detail="Missing query parameter")

        return StreamingResponse(
            proccess_stream_response(query, session_id, web_search, use_rag),
            media_type='text/event-stream'
        )
    except Exception as e:
        logger.error(f"流式响应端点错误: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"detail": "Internal server error in stream endpoint"})

@db_retry(max_retries=3, delay=1)
async def save_message_with_retry(session_id, role, content):
    """带重试的消息保存函数"""
    async with AsyncSessionLocal() as db_session:
        message = ChatMessage(
            id=str(uuid.uuid4()),
            session_id=session_id,
            role=role,
            content=content,
            created_at=datetime.utcnow()
        )
        db_session.add(message)
        await db_session.commit()

async def proccess_stream_response(query: str, session_id: Optional[str], web_search: bool, use_rag: bool = True) -> AsyncGenerator[str, None]:
    """
    优化的流式响应处理，使用多个短会话避免连接超时
    """
    global AsyncSessionLocal, rag_system
    
    # 1. 预先处理所有数据库操作
    chat_session_obj = None
    history_for_llm = []
    current_session_id = None
    
    try:
        # 在独立的数据库会话中完成所有初始化操作
        async with AsyncSessionLocal() as db_session:
            # 处理会话查找和创建
            if session_id:
                stmt = (
                    select(ChatSession)
                    .where(ChatSession.id == session_id)
                    .options(selectinload(ChatSession.messages))
                )
                result = await db_session.execute(stmt)
                chat_session_obj = result.scalars().first()
            
            if not chat_session_obj:
                effective_session_id = str(uuid.uuid4())
                chat_session_obj = ChatSession(
                    id=effective_session_id, 
                    user_id="default_user", 
                    created_at=datetime.utcnow()
                )
                db_session.add(chat_session_obj)
                await db_session.flush()
                await db_session.commit()  # 立即提交
                current_session_id = effective_session_id
                
                # 通知客户端新会话ID
                new_session_event = {"type": "session_id", "id": current_session_id}
                yield f"data: {json.dumps(new_session_event)}\n\n"
                logger.info(f"创建了新的聊天会话: {current_session_id}")
            else:
                current_session_id = chat_session_obj.id
                # 加载历史消息
                if chat_session_obj.messages:
                    materialized_orm_messages = list(chat_session_obj.messages)
                    if materialized_orm_messages:
                        raw_messages_data = []
                        for msg_orm in materialized_orm_messages:
                            raw_messages_data.append({
                                "role": msg_orm.role,
                                "content": msg_orm.content,
                                "created_at": msg_orm.created_at
                            })
                        
                        sorted_messages_data = sorted(raw_messages_data, key=lambda m: m["created_at"])
                        for msg_data in sorted_messages_data:
                            history_for_llm.append({"role": msg_data["role"], "content": msg_data["content"]})
                        logger.info(f"成功为会话 {current_session_id} 加载了 {len(history_for_llm)} 条历史消息")
        
        # 2. 在独立会话中保存用户消息
        await save_message_with_retry(current_session_id, "user", query)
        
        # 3. RAG检索逻辑（无数据库操作）
        rag_context = None
        rag_sources = []
        
        if use_rag and rag_system:
            try:
                logger.info(f"开始RAG检索，查询: '{query}'")
                # 检查RAG系统状态
                stats = rag_system.get_stats()
                if stats['total_documents'] == 0:
                    logger.warning("知识库为空，跳过RAG检索")
                    rag_info_event = {
                        "type": "rag_info", 
                        "sources": [],
                        "has_context": False,
                        "message": "知识库为空，请先添加文档"
                    }
                    yield f"data: {json.dumps(rag_info_event)}\n\n"
                else:
                    logger.info(f"开始RAG检索 (会话: {current_session_id})")
                    # 支持重排参数和智能相关性判断
                    rag_result = await rag_system.answer_question(
                        query, 
                        top_k=3,  # 从5减少到3
                        use_rerank=True,
                        rerank_strategy='hybrid',
                        relevance_threshold=0.12,  # 从0.25调整回0.12
                        enable_smart_rag=True  # 启用智能RAG
                    )
                    
                    # 检查是否需要使用RAG结果
                    if rag_result and rag_result.get('has_context') and rag_result.get('context'):
                        rag_context = rag_result['context']
                        rag_sources = rag_result.get('sources', [])
                        
                        logger.info(f"RAG检索成功，找到 {len(rag_sources)} 个相关文档")
                        logger.info(f"重排策略: {rag_result.get('rerank_strategy', 'none')}")
                        logger.debug(f"RAG上下文长度: {len(rag_context)}")
                        
                        # 发送RAG信息给前端
                        rag_info_event = {
                            "type": "rag_info", 
                            "sources": rag_sources,
                            "has_context": True,
                            "context_length": len(rag_context),
                            "relevance_check": rag_result.get('relevance_check', 'passed')
                        }
                        yield f"data: {json.dumps(rag_info_event)}\n\n"
                    else:
                        # 问题与知识库不相关，记录日志但不发送RAG信息
                        relevance_status = rag_result.get('relevance_check', 'unknown')
                        logger.info(f"问题与知识库不相关 ({relevance_status})，将直接使用大模型回答")
                        
                        # 可选：发送提示信息给前端
                        rag_info_event = {
                            "type": "rag_info", 
                            "sources": [],
                            "has_context": False,
                            "message": "问题与知识库内容不相关，使用通用AI回答",
                            "relevance_check": relevance_status
                        }
                        yield f"data: {json.dumps(rag_info_event)}\n\n"
                    
            except Exception as e:
                logger.error(f"RAG检索失败: {e}", exc_info=True)
                rag_error_event = {
                    "type": "rag_error", 
                    "error": str(e)
                }
                yield f"data: {json.dumps(rag_error_event)}\n\n"
        
        # 4. 联网搜索逻辑（如果需要）
        search_context_str = None
        if web_search:
            # 此处应为实际的联网搜索逻辑
            # search_results = await perform_web_search(query) 
            # search_context_str = format_search_results_for_llm(search_results)
            search_context_str = "模拟的联网搜索上下文信息。" # 当前为占位符
            web_search_info_event = {"type": "info", "message": "正在进行联网搜索（模拟）..."}
            yield f"data: {json.dumps(web_search_info_event)}\n\n"
            # 实际应用中，联网搜索结果会通过 search_context_str 传递给LLM
        
        # 5. LLM流式响应处理
        full_llm_response = ""
        try:
            # 调用增强的LLM响应生成器
            async for llm_chunk_data in generate_llm_response_with_rag(
                query, history_for_llm, search_context_str, rag_context, current_session_id
            ):
                logger.info(f"收到LLM响应块 (会话: {current_session_id}): {llm_chunk_data}")
                if "error" in llm_chunk_data:
                    error_message = llm_chunk_data["error"]
                    logger.error(f"LLM 生成错误 (会话: {current_session_id}): {error_message}")
                    error_event = {"type": "error", "message": f"LLM 错误: {error_message}"}
                    yield f"data: {json.dumps(error_event)}\n\n"
                    full_llm_response = f"LLM 错误: {error_message}" # 记录错误信息
                    break # 遇到LLM错误则停止处理后续块
                
                content_piece = llm_chunk_data.get("content", "")
                if content_piece:
                    logger.info(f"处理LLM响应块 (会话: {current_session_id}): {content_piece}")
                    full_llm_response += content_piece
                    chunk_event = {"type": "message_chunk", "content": content_piece}
                    yield f"data: {json.dumps(chunk_event)}\n\n"
            
        except Exception as e:
            logger.error(f"LLM响应生成失败: {e}", exc_info=True)
            error_response = "抱歉，生成回复时出现错误"
            full_llm_response = error_response
            yield f"data: {json.dumps({'type': 'error', 'content': error_response})}\n\n"
        
        # 6. 在独立会话中保存助手响应
        assistant_message_content = full_llm_response.strip()
        if not assistant_message_content and not ("LLM 错误" in full_llm_response):
            assistant_message_content = "抱歉，我暂时无法回答这个问题。"
        
        if assistant_message_content:
            await save_message_with_retry(current_session_id, "assistant", assistant_message_content)
        
        # 7. 发送完成信号
        yield f"data: {json.dumps({'type': 'done'})}\n\n"
        
    except Exception as e:
        logger.error(f"proccess_stream_response 中的错误: {e}", exc_info=True)
        yield f"data: {json.dumps({'type': 'error', 'content': '处理请求时发生错误'})}\n\n"

async def save_chat_messages(db, session_id, user_message, assistant_message):
    """保存对话消息到数据库"""
    try:
        messages = [
            ChatMessage(
                session_id=session_id,
                role="user",
                content=user_message,
                timestamp=datetime.utcnow()
            ),
            ChatMessage(
                session_id=session_id,
                role="assistant",
                content=assistant_message,
                timestamp=datetime.utcnow()
            )
        ]
        db.add_all(messages)
        await db.commit()
    except Exception as e:
        logger.error(f"保存对话记录失败: {str(e)}")
        await db.rollback()
        raise

# 新增增强的LLM响应生成函数
async def generate_llm_response_with_rag(query, history, search_context, rag_context, session_id):
    """生成RAG增强回答的异步生成器"""
    try:
        # 设置最大长度限制（留出安全边距）
        MAX_PROMPT_LENGTH = 25000  # 比API限制30720小一些
        
        # 构建上下文
        context_parts = []
        
        # 添加RAG上下文（优先级最高）
        if rag_context:
            # 限制RAG上下文长度
            if len(rag_context) > 15000:
                rag_context = rag_context[:15000] + "\n\n[内容已截断...]"
            context_parts.append(f"知识库相关内容:\n{rag_context}")
        
        # 添加搜索上下文
        if search_context:
            if len(search_context) > 5000:
                search_context = search_context[:5000] + "\n\n[内容已截断...]"
            context_parts.append(search_context)
        
        # 添加历史对话（限制条数和长度）
        if history:
            # 只保留最近3条对话，每条限制200字符
            recent_history = history[-3:]
            formatted_history_parts = []
            for msg in recent_history:
                content = msg['content']
                if len(content) > 200:
                    content = content[:200] + "..."
                formatted_history_parts.append(f"{'用户' if msg['role'] == 'user' else '助手'}: {content}")
            
            history_text = "\n".join(formatted_history_parts)
            context_parts.append(f"历史对话:\n{history_text}")

        # 构建增强的提示词
        if rag_context:
            # 计算当前长度
            base_prompt = f"""以下是相关的参考资料：

当前用户问题: {query}

请主要针对当前用户问题进行回答。优先使用参考资料中的信息，如有需要可结合你的知识进行补充。"""
            
            context_text = "\n\n".join(context_parts)
            full_prompt = base_prompt.replace("以下是相关的参考资料：", f"以下是相关的参考资料：\n\n{context_text}")
            
            # 检查总长度并截断
            if len(full_prompt) > MAX_PROMPT_LENGTH:
                # 重新构建，优先保留RAG上下文
                available_length = MAX_PROMPT_LENGTH - len(base_prompt) - 100  # 留出缓冲
                if len(rag_context) > available_length:
                    rag_context = rag_context[:available_length] + "\n\n[内容已截断...]"
                
                formatted_prompt = f"""以下是相关的参考资料：

{rag_context}

当前用户问题: {query}

请主要针对当前用户问题进行回答。优先使用参考资料中的信息，如有需要可结合你的知识进行补充。"""
            else:
                formatted_prompt = full_prompt
        else:
            # 没有RAG上下文时使用原有逻辑
            formatted_prompt = f"""当前用户问题: {query}

请针对用户的问题提供准确、有帮助的回答。"""
        
        # 最终长度检查
        if len(formatted_prompt) > MAX_PROMPT_LENGTH:
            formatted_prompt = formatted_prompt[:MAX_PROMPT_LENGTH] + "\n\n[提示词已截断...]"
        
        logger.info(f"最终prompt长度: {len(formatted_prompt)} 字符")
        logger.debug(f"LLM Prompt for session {session_id}:\n{formatted_prompt}")

        # 调用原有的LLM生成逻辑
        async for chunk in generate_llm_response(formatted_prompt, [], None, session_id):
            yield chunk
            
    except Exception as e:
        logger.error(f"生成RAG增强回答失败 (会话: {session_id}): {str(e)}", exc_info=True)
        yield {"error": f"生成回答时发生内部错误: {str(e)}"}


async def generate_llm_response(query, history, search_context, session_id):
    """生成LLM响应的异步生成器"""
    try:
        # 构建上下文
        context_parts = []
        if search_context:
            context_parts.append(search_context)
        
        # 始终添加历史对话（如果存在）
        if history:
            formatted_history = "\n".join([
                f"{'用户' if msg['role'] == 'user' else '助手'}: {msg['content']}"
                for msg in history[-5:]  # 保留最近5条对话
            ])
            context_parts.append(f"历史对话:\n{formatted_history}")

        # 修改后的提示词 - 强调以当前问题为主
        if context_parts:
            full_context_str = "\n\n".join(filter(None, context_parts))
            formatted_prompt = f"""
参考信息:

{full_context_str}

当前用户问题: {query}

请主要针对当前用户问题进行回答。如果当前问题是独立的（如问候、感谢等），请直接回答，无需参考历史对话。如果当前问题与历史对话相关，则可以适当参考历史内容。如果使用了网络搜索信息，请注明来源。
        """
        else:
            # 没有任何上下文时的简洁提示
            formatted_prompt = f"""
当前用户问题: {query}

请针对用户的问题提供准确、有帮助的回答。
        """
        logger.debug(f"LLM Prompt for session {session_id}:\n{formatted_prompt}")

        # 调用模型生成回答 - 修改这部分
        model_name = "qwen-max"  # 设置默认模型名称
        generation = dashscope.Generation()
        
        logger.info(f"调用DashScope API (会话: {session_id}) 进行LLM响应生成...")
        # 使用同步方式调用API，不使用await
        response = generation.call(
            model=model_name,
            prompt=formatted_prompt,
            stream=True,
            result_format='message',  # 添加结果格式参数
            api_key=os.getenv('DASHSCOPE_API_KEY')
        )
        
        logger.info(f"DashScope API调用完成 (会话: {session_id})")


        # 处理流式响应
        # 在generate_llm_response函数中改进响应处理
        for chunk in response:
            logger.debug(f"原始响应块: {chunk}")
            
            try:
                # 检查响应状态
                if hasattr(chunk, 'status_code') and chunk.status_code == 200:
                    if hasattr(chunk, 'output') and chunk.output:
                        output = chunk.output
                        content = None
                        
                        # 尝试多种路径提取内容
                        if isinstance(output, dict):
                            # 路径1: choices.message.content
                            if 'choices' in output and output['choices']:
                                choice = output['choices'][0]
                                if 'message' in choice and 'content' in choice['message']:
                                    content = choice['message']['content']
                            
                            # 路径2: text字段
                            if not content and 'text' in output:
                                content = output['text']
                            
                            # 路径3: 直接的content字段
                            if not content and 'content' in output:
                                content = output['content']
                    
                    if content and content.strip():
                        logger.info(f"提取到内容: {content}")
                        yield {"content": content}
                    else:
                        logger.warning(f"无法提取内容，output结构: {output}")
                else:
                    # 处理错误响应
                    error_code = getattr(chunk, 'code', 'unknown')
                    error_message = getattr(chunk, 'message', 'unknown error')
                    logger.error(f"API错误: {error_code} - {error_message}")
                    yield {"error": f"API错误: {error_code} - {error_message}"}
                    break
            except Exception as e:
                logger.error(f"处理响应块时出错: {e}")
                yield {"error": f"处理响应时出错: {str(e)}"}
                break

    except Exception as e:
        logger.error(f"生成回答失败 (会话: {session_id}): {str(e)}", exc_info=True)
        yield {"error": f"生成回答时发生内部错误: {str(e)}"}

# 处理文件上传请求
@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        #检查文件类型
        if not doc_processor.is_supported(file.filename):
            raise HTTPException(
                status_code=400, 
                detail=f"不支持的文件类型。支持的类型: {list(doc_processor.supported_types.keys())}"
            )
        
        # 确保文件目录存在,不存在则创建
        upload_dir = "E:/code/AIProjectCode/trae_code/project/RAG-EKB/backend/data/uploads"
        os.makedirs(upload_dir, exist_ok=True)

        #保存文件
        file_path = os.path.join(upload_dir, file.filename)

        #写入到文件夹中
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # 保存文件后，将文档信息存入数据库
        async with db_manager.get_session() as session:
            # 检查文档是否已存在
            existing_doc = await session.execute(
                select(Document).where(Document.filename == file.filename)
            )
            if existing_doc.scalar_one_or_none():
                raise HTTPException(status_code=400, detail="文档已存在")
            
            # 创建文档记录
            document = Document(
                filename=file.filename,
                original_name=file.filename,
                file_path=file_path,
                file_size=len(content),
                file_type=os.path.splitext(file.filename)[1],
                processing_status="uploaded"
            )
            
            session.add(document)
            await session.commit()
            
            # 刷新对象以获取数据库生成的ID
            await session.refresh(document)
            
            return {
                "message": "文件上传成功",
                "document_id": document.id,
                "filename": file.filename,
                "size": len(content),
                "status": "uploaded"
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"文件上传异常: {str(e)}")
        logger.error(f"堆栈跟踪:\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"上传失败: {str(e)}")

'''
实现文档列表接口
前端界面显示文件
'''
@app.get("/api/documents")
async def get_documents():
    """获取文档列表"""
    try:
        async with db_manager.get_session() as session:
            result = await session.execute(select(Document).order_by(Document.created_at.desc()))
            documents = result.scalars().all()
            
            return [
                {
                    "id": doc.id,
                    "name": doc.filename,  # 改为name字段
                    "filename": doc.filename,
                    "original_name": doc.original_name,
                    "size": doc.file_size,  # 改为size字段
                    "file_size": doc.file_size,
                    "file_type": doc.file_type,
                    "processing_status": doc.processing_status,
                    "processed_at": doc.processed_at.isoformat() if doc.processed_at else None,
                    "chunk_count": doc.chunk_count,
                    "processing_time": doc.processing_time,
                    "created_at": doc.created_at.isoformat(),
                    "error_message": doc.error_message
                }
                for doc in documents
            ]
    except Exception as e:
        logger.error(f"获取文档列表异常: {str(e)}")
        logger.error(f"堆栈跟踪:\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"获取文档列表失败: {str(e)}")

'''
实现删除文档接口
功能： 根据文档ID删除文档
'''
@app.delete("/api/documents/{document_id}")
async def delete_document(document_id: str):
    """删除文档"""
    try:
        async with db_manager.get_session() as session:
            # 查找文档记录
            result = await session.execute(select(Document).where(Document.id == document_id))
            document = result.scalar_one_or_none()
            
            if not document:
                raise HTTPException(status_code=404, detail=f"文档 {document_id} 不存在")
            
            # 删除物理文件
            if os.path.exists(document.file_path):
                os.remove(document.file_path)
            
            # 删除数据库记录
            await session.delete(document)
            await session.commit()
            
            return {"message": f"文档 {document.filename} 删除成功"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除文档失败: {str(e)}")

# 新增会话管理API
@app.post("/api/chat/sessions")
async def create_chat_session(user_id: str = "default_user"):
    """创建新的聊天会话"""
    try:
        async with db_manager.get_session() as session:
            new_session = ChatSession(
                id=str(uuid.uuid4()),
                user_id=user_id,
                created_at=datetime.utcnow()
            )
            session.add(new_session)
            await session.commit()
            
            return {
                "session_id": new_session.id,
                "user_id": new_session.user_id,
                "created_at": new_session.created_at.isoformat()
            }
    except Exception as e:
        logger.error(f"创建聊天会话失败: {str(e)}")
        raise HTTPException(status_code=500, detail="创建聊天会话失败")

@app.delete("/api/chat/sessions/{session_id}")
async def delete_chat_session(session_id: str):
    """删除聊天会话及其所有消息"""
    try:
        async with db_manager.get_session() as session:
            # 删除会话的所有消息
            await session.execute(
                select(ChatMessage).where(ChatMessage.session_id == session_id)
            )
            
            # 删除会话
            result = await session.execute(
                select(ChatSession).where(ChatSession.id == session_id)
            )
            chat_session = result.scalar_one_or_none()
            
            if not chat_session:
                raise HTTPException(status_code=404, detail="会话不存在")
            
            await session.delete(chat_session)
            await session.commit()
            
            return {"message": "会话删除成功"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除聊天会话失败: {str(e)}")
        raise HTTPException(status_code=500, detail="删除聊天会话失败")

# 添加RAG相关的API端点
@app.post("/api/rag/add_document")
async def add_document_to_rag(file: UploadFile = File(...)):
    """添加文档到RAG知识库"""
    try:
        # 保存上传的文件
        upload_dir = "E:/code/AIProjectCode/trae_code/project/RAG-EKB/backend/data/uploads"
        os.makedirs(upload_dir, exist_ok=True)
        
        file_path = os.path.join(upload_dir, file.filename)
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # 保存文档信息到数据库
        async with db_manager.get_session() as session:
            # 检查文档是否已存在
            existing_doc = await session.execute(
                select(Document).where(Document.filename == file.filename)
            )
            if existing_doc.scalar_one_or_none():
                raise HTTPException(status_code=400, detail="文档已存在")
            
            # 创建文档记录
            document = Document(
                filename=file.filename,
                original_name=file.filename,
                file_path=file_path,
                file_size=len(content),
                file_type=os.path.splitext(file.filename)[1],
                processing_status="processing"
            )
            
            session.add(document)
            await session.flush()  # 获取document.id
            
            try:
                # 添加到RAG系统
                result = await rag_system.add_document(file_path, file.filename)
                
                # 更新处理状态
                document.processing_status = "completed"
                document.processed_at = datetime.utcnow()
                document.chunk_count = result.get('chunk_count', 0)
                document.processing_time = result.get('processing_time', 0)
                
                await session.commit()
                
                return {
                    "status": "success",
                    "message": "文档已成功添加到知识库",
                    "document_id": document.id,
                    "result": result
                }
                
            except Exception as e:
                # 处理失败，更新状态
                document.processing_status = "failed"
                document.error_message = str(e)
                await session.commit()
                raise e
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"添加文档到RAG失败: {e}")
        raise HTTPException(status_code=500, detail=f"添加文档失败: {str(e)}")

@app.get("/api/rag/stats")
async def get_rag_stats():
    """获取RAG知识库统计信息"""
    try:
        logger.info("收到RAG统计信息请求")
        
        # 详细检查rag_system状态
        if rag_system is None:
            logger.error("RAG系统为None，可能初始化失败")
            return {
                "status": "error", 
                "message": "RAG系统未初始化",
                "details": "rag_system变量为None，请检查服务启动日志"
            }
        
        logger.info("RAG系统存在，正在获取统计信息...")
        stats = rag_system.get_stats()
        logger.info(f"获取到统计信息: {stats}")
        
        return {
            "status": "success", 
            "stats": stats,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"获取RAG统计信息失败: {e}", exc_info=True)
        return {
            "status": "error",
            "message": f"获取统计信息失败: {str(e)}",
            "error_type": type(e).__name__
        }

@app.delete("/api/rag/clear")
async def clear_rag_knowledge_base():
    """清空RAG知识库"""
    try:
        if rag_system:
            rag_system.clear_knowledge_base()
            return {"status": "success", "message": "知识库已清空"}
        else:
            return {"status": "error", "message": "RAG系统未初始化"}
    except Exception as e:
        logger.error(f"清空知识库失败: {e}")
        raise HTTPException(status_code=500, detail=f"清空知识库失败: {str(e)}")

# 日志中间件，记录请求日志  
@app.middleware("http")
async def log_requests(request, call_next):

    logger.info(f"收到请求: {request.method} {request.url}")
    response = await call_next(request)
    return response

# 日志相关模型
class LogEntry(BaseModel):
    timestamp: str
    level: str
    module: str
    message: str
    tags: List[str] = []

class LogSaveRequest(BaseModel):
    path: str
    logs: List[LogEntry]

class LogInitRequest(BaseModel):
    path: str

class LogSearchRequest(BaseModel):
    path: str
    date: Optional[str] = None
    tags: Optional[List[str]] = None
    level: Optional[str] = None
    module: Optional[str] = None
    query: Optional[str] = None
    limit: Optional[int] = 100

# 初始化日志目录
@app.post("/api/logs/init")
async def init_log_directory(request: LogInitRequest):
    try:
        os.makedirs(request.path, exist_ok=True)
        return {"success": True, "message": "日志目录初始化成功"}
    except Exception as e:
        logger.error(f"初始化日志目录失败: {str(e)}")
        return {"success": False, "error": str(e)}

# 保存日志到文件
@app.post("/api/logs/save")
async def save_logs(request: LogSaveRequest):
    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(request.path), exist_ok=True)
        
        # 将日志追加到文件
        mode = 'a' if os.path.exists(request.path) else 'w'
        with open(request.path, mode, encoding='utf-8') as f:
            for log in request.logs:
                f.write(json.dumps(log.dict(), ensure_ascii=False) + '\n')
        
        return {"success": True, "count": len(request.logs)}
    except Exception as e:
        logger.error(f"保存日志失败: {str(e)}")
        return {"success": False, "error": str(e)}

# 搜索日志
@app.post("/api/logs/search")
async def search_logs(request: LogSearchRequest):
    try:
        results = []
        
        # 确定要搜索的文件
        log_file = request.path
        if request.date:
            log_file = os.path.join(request.path, f"{request.date}.log")
        
        # 如果文件不存在，返回空结果
        if not os.path.exists(log_file):
            return {"logs": [], "count": 0}
        
        # 读取并过滤日志
        with open(log_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    log = json.loads(line.strip())
                    
                    # 应用过滤条件
                    if request.level and log.get('level') != request.level:
                        continue
                        
                    if request.module and log.get('module') != request.module:
                        continue
                        
                    if request.tags:
                        log_tags = set(log.get('tags', []))
                        if not all(tag in log_tags for tag in request.tags):
                            continue
                    
                    if request.query and request.query.lower() not in log.get('message', '').lower():
                        continue
                        
                    results.append(log)
                    
                    # 限制结果数量
                    if len(results) >= request.limit:
                        break
                        
                except json.JSONDecodeError:
                    continue
        
        return {"logs": results, "count": len(results)}
    except Exception as e:
        logger.error(f"搜索日志失败: {str(e)}")
        return {"success": False, "error": str(e), "logs": []}

# 运行后端服务
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# 在现有的API端点后添加

# 文档处理配置相关的数据模型
from pydantic import BaseModel, Field

# 定义详细的配置模型
class ChunkingConfig(BaseModel):
    method: str = "fixed_size"
    chunk_size: int = 1000
    overlap_size: int = 200
    custom_pattern: str = ""
    min_chunk_size: int = 100
    merge_small_chunks: bool = True
    preserve_headers: bool = True
    preserve_tables: bool = True
    include_metadata: bool = True
    separator: str = "double_newline"
    custom_separator: str = ""
    preserve_structure: bool = True
    extract_images: bool = False
    extract_tables: bool = True
    parse_method: str = "auto"
    enable_formula: bool = True
    enable_table: bool = True
    enable_ocr: bool = False
    device_mode: str = "cpu"
    lang: str = "zh"
    respect_headers: bool = True
    preserve_code_blocks: bool = True
    separators: List[str] = Field(default_factory=lambda: ['\n\n', '\n', '.', '!', '?'])
    
    class Config:
        extra = "allow"

class EmbeddingConfig(BaseModel):
    model: str = "text-embedding-ada-002"
    dimensions: int = 1536
    batchSize: int = 100
    
    class Config:
        extra = "allow"

class IndexingConfig(BaseModel):
    vectorStore: str = "faiss"
    similarity: str = "cosine"
    indexType: str = "flat"
    nlist: int = 100
    similarity_threshold: float = 0.7
    
    class Config:
        extra = "allow"

class PreprocessingConfig(BaseModel):
    removeStopwords: bool = True
    lowercase: bool = True
    removeSpecialChars: bool = False
    remove_duplicates: bool = True
    clean_text: bool = True
    extract_metadata: bool = True
    language_detection: bool = False
    
    class Config:
        extra = "allow"

class ProcessingConfig(BaseModel):
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    indexing: IndexingConfig = Field(default_factory=IndexingConfig)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    
    class Config:
        extra = "allow"

class DocumentProcessRequest(BaseModel):
    document_id: Union[str, int]  # 允许字符串或整数
    config: ProcessingConfig
    
    class Config:
        # 允许额外字段
        extra = "allow"

# 批量文档处理请求模型
class BatchDocumentProcessRequest(BaseModel):
    document_ids: List[Union[str, int]]
    config: Dict[str, Any] = {}
    
    @validator('document_ids', pre=True)
    def convert_document_ids(cls, v):
        # 将所有ID转换为字符串
        return [str(id) for id in v]
    
    class Config:
        extra = "allow"

# 文档处理配置API
@app.post("/api/process_document")
async def process_document_with_config(request: DocumentProcessRequest):
    """根据用户配置处理文档"""
    try:
        async with db_manager.get_session() as session:
            # 确保 document_id 是字符串类型
            document_id = str(request.document_id)
            
            # 获取文档记录
            result = await session.execute(
                select(Document).where(Document.id == document_id)
            )
            document = result.scalar_one_or_none()
            
            if not document:
                raise HTTPException(status_code=404, detail="文档不存在")
            
            # 立即获取所有需要的属性，避免在异步处理中延迟加载
            file_path = document.file_path
            doc_id = document.id  # 提前获取ID
            
            # 更新处理状态
            document.processing_status = "processing"
            document.processing_config = json.dumps(request.config.dict())
            await session.commit()
            
            try:
                # 执行文档处理
                start_time = time.time()
                
                # 使用已获取的file_path，避免在异步处理中访问ORM对象
                processor = ConfigurableDocumentProcessor(request.config.dict())
                result = await processor.process_document_async(file_path)
                
                processing_time = time.time() - start_time
                
                # 更新处理结果
                document.processing_status = "processed"
                document.processed_at = datetime.utcnow()
                chunk_count = result.get('chunk_count', 0)  # 直接获取值
                document.chunk_count = chunk_count
                document.vector_dimension = result.get('vector_dimension', 0)
                document.index_size = result.get('index_size', 0)
                document.processing_time = processing_time
                document.word_count = result.get('word_count', 0)
                document.error_message = None
                
                await session.commit()
                
                # 使用直接获取的值，避免访问ORM对象
                return {
                    "message": "文档处理成功",
                    "document_id": doc_id,  # 使用提前获取的ID
                    "processing_time": processing_time,
                    "chunk_count": chunk_count  # 使用直接获取的值
                }
                
            except Exception as e:
                # 处理失败，更新状态
                logger.error(f"文档处理异常: {str(e)}")
                logger.error(f"堆栈跟踪:\n{traceback.format_exc()}")
                document.processing_status = "failed"
                document.error_message = str(e)
                await session.commit()
                raise HTTPException(status_code=500, detail=f"文档处理失败: {str(e)}")
                
    except Exception as e:
        logger.error(f"处理请求异常: {str(e)}")
        logger.error(f"堆栈跟踪:\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"处理请求失败: {str(e)}")

# 批量文档处理接口
@app.post("/api/process_documents")
async def process_documents_batch(request: BatchDocumentProcessRequest):
    """批量处理多个文档"""
    try:
        # 添加更详细的调试日志
        logger.info(f"收到批量处理请求: document_ids={request.document_ids}")
        logger.info(f"原始配置: {request.config}")
        
        # 确保配置是字典格式
        if request.config is None:
            config_dict = {}
        elif isinstance(request.config, dict):
            config_dict = request.config
        else:
            # 如果不是字典，尝试转换
            try:
                config_dict = dict(request.config)
            except:
                config_dict = {}
        
        logger.info(f"处理后的配置: {config_dict}")
        
        # 预加载文档数据，避免在会话外访问
        document_data = []
        
        async with db_manager.get_session() as session:
            # 验证所有文档是否存在
            document_ids = [str(doc_id) for doc_id in request.document_ids]
            
            result = await session.execute(
                select(Document).where(Document.id.in_(document_ids))
            )
            documents = result.scalars().all()
            
            if len(documents) != len(document_ids):
                missing_ids = set(document_ids) - {doc.id for doc in documents}
                raise HTTPException(
                    status_code=404, 
                    detail=f"以下文档不存在: {list(missing_ids)}"
                )
            
            # 预加载所有需要的文档数据
            for doc in documents:
                document_data.append({
                    'id': doc.id,
                    'filename': doc.filename,  # 在会话内访问
                    'document_obj': doc
                })
            
            # 使用处理后的配置字典
            config_json = json.dumps(config_dict, ensure_ascii=False)
            
            # 更新所有文档状态为处理中
            for document in documents:
                document.processing_status = "processing"
                document.processing_config = config_json
            
            await session.commit()
        
        # 在会话外进行文档处理
        results = []
        total_processing_time = 0
        
        try:
            # 使用配置字典创建处理器
            processor = ConfigurableDocumentProcessor(config_dict)
            
            for doc_data in document_data:
                start_time = time.time()
                
                try:
                    # 处理单个文档
                    file_path = os.path.join(UPLOAD_DIR, doc_data['filename'])
                    result = await processor.process_document_async(file_path)
                    
                    processing_time = time.time() - start_time
                    total_processing_time += processing_time
                    
                    results.append({
                        "document_id": doc_data['id'],
                        "filename": doc_data['filename'],
                        "status": "success",
                        "processing_time": processing_time,
                        "chunk_count": result.get('chunk_count', 0),
                        "vector_count": result.get('vector_count', 0)
                    })
                    
                    # 在新的会话中更新文档状态
                    async with db_manager.get_session() as session:
                        doc = await session.get(Document, doc_data['id'])
                        if doc:
                            doc.processing_status = "processed"
                            doc.processing_time = processing_time
                            doc.chunk_count = result.get('chunk_count', 0)
                            doc.vector_count = result.get('vector_count', 0)
                            doc.error_message = None
                            await session.commit()
                    
                except Exception as doc_error:
                    logger.error(f"处理文档 {doc_data['filename']} 失败: {str(doc_error)}")
                    
                    results.append({
                        "document_id": doc_data['id'],
                        "filename": doc_data['filename'],
                        "status": "failed",
                        "error": str(doc_error)
                    })
                    
                    # 在新的会话中更新文档状态为失败
                    async with db_manager.get_session() as session:
                        doc = await session.get(Document, doc_data['id'])
                        if doc:
                            doc.processing_status = "failed"
                            doc.error_message = str(doc_error)
                            await session.commit()
            
            # 返回处理结果摘要
            success_count = len([r for r in results if r["status"] == "success"])
            failed_count = len([r for r in results if r["status"] == "failed"])
            
            return {
                "success": True,
                "total_documents": len(document_ids),
                "success_count": success_count,
                "failed_count": failed_count,
                "total_processing_time": total_processing_time,
                "results": results
            }
            
        except Exception as e:
            # 如果整个批量处理失败，更新所有文档状态
            async with db_manager.get_session() as session:
                for doc_data in document_data:
                    doc = await session.get(Document, doc_data['id'])
                    if doc:
                        doc.processing_status = "failed"
                        doc.error_message = str(e)
                await session.commit()
            raise e
                
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"批量处理请求异常: {str(e)}")
        logger.error(f"堆栈跟踪:\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"批量处理失败: {str(e)}")

# 获取处理预设配置
@app.get("/api/processing_presets")
async def get_processing_presets():
    """
    获取预设的处理配置
    """
    presets = {
        "default": {
            "name": "默认配置",
            "description": "平衡的处理策略，适合大多数文档",
            "config": {
                "chunking": {
                    "method": "fixed_size",
                    "chunk_size": 500,
                    "overlap_size": 50
                },
                "embedding": {
                    "model": "text-embedding-v3",
                    "batch_size": 10
                },
                "indexing": {
                    "type": "faiss_flat",
                    "similarity_threshold": 0.7
                },
                "preprocessing": {
                    "remove_duplicates": True,
                    "clean_text": True,
                    "extract_metadata": True,
                    "language_detection": False
                }
            }
        },
        "fast": {
            "name": "快速处理",
            "description": "优化速度，适合大量文档的快速处理",
            "config": {
                "chunking": {
                    "method": "fixed_size",
                    "chunk_size": 800,
                    "overlap_size": 80
                },
                "embedding": {
                    "model": "all-MiniLM-L6-v2",
                    "batch_size": 20
                },
                "indexing": {
                    "type": "faiss_ivf",
                    "nlist": 50,
                    "similarity_threshold": 0.6
                },
                "preprocessing": {
                    "remove_duplicates": False,
                    "clean_text": True,
                    "extract_metadata": False,
                    "language_detection": False
                }
            }
        },
        "accurate": {
            "name": "精确处理",
            "description": "优化准确性，适合重要文档的精细处理",
            "config": {
                "chunking": {
                    "method": "semantic",
                    "chunk_size": 300,
                    "overlap_size": 30
                },
                "embedding": {
                    "model": "bge-large-zh",
                    "batch_size": 5
                },
                "indexing": {
                    "type": "faiss_hnsw",
                    "similarity_threshold": 0.8
                },
                "preprocessing": {
                    "remove_duplicates": True,
                    "clean_text": True,
                    "extract_metadata": True,
                    "language_detection": True
                }
            }
        }
    }
    
    return presets





# MCP工具配置
MCP_TOOLS = {
    "weather": {
        "name": "天气查询",
        "description": "查询指定城市的天气信息",
        "parameters": {
            "city": {"type": "string", "description": "城市名称"}
        }
    },
    "calculator": {
        "name": "计算器",
        "description": "执行数学计算",
        "parameters": {
            "expression": {"type": "string", "description": "数学表达式"}
        }
    },
    "web_search": {
        "name": "网络搜索",
        "description": "搜索网络信息",
        "parameters": {
            "query": {"type": "string", "description": "搜索关键词"}
        }
    }
}

# 获取MCP工具列表
@app.get("/api/mcp/tools")
async def get_mcp_tools():
    """获取可用的MCP工具列表"""
    return {"tools": MCP_TOOLS}

# 执行MCP工具
@app.post("/api/mcp/execute")
async def execute_mcp_tool(request: Dict[str, Any]):
    """执行MCP工具"""
    try:
        tool_name = request.get("tool")
        parameters = request.get("parameters", {})
        
        if tool_name not in MCP_TOOLS:
            raise HTTPException(status_code=400, detail=f"未知的工具: {tool_name}")
        
        # 根据工具类型执行相应的逻辑
        if tool_name == "weather":
            result = await execute_weather_tool(parameters)
        elif tool_name == "calculator":
            result = await execute_calculator_tool(parameters)
        elif tool_name == "web_search":
            result = await execute_web_search_tool(parameters)
        else:
            result = {"error": "工具未实现"}
        
        return {"success": True, "result": result}
    
    except Exception as e:
        logger.error(f"MCP工具执行失败: {str(e)}")
        return {"success": False, "error": str(e)}

# MCP工具实现函数
async def execute_weather_tool(params: Dict[str, Any]):
    """执行天气查询工具"""
    city = params.get("city", "北京")
    # 这里可以集成真实的天气API
    return {
        "city": city,
        "temperature": "22°C",
        "weather": "晴天",
        "humidity": "65%"
    }

async def execute_calculator_tool(params: Dict[str, Any]):
    """执行计算器工具"""
    expression = params.get("expression", "")
    try:
        # 安全的数学表达式计算
        result = eval(expression, {"__builtins__": {}}, {})
        return {"expression": expression, "result": result}
    except Exception as e:
        return {"error": f"计算错误: {str(e)}"}

async def execute_web_search_tool(params: Dict[str, Any]):
    """执行网络搜索工具"""
    query = params.get("query", "")
    # 这里可以集成搜索引擎API
    return {
        "query": query,
        "results": [
            {"title": "搜索结果1", "url": "https://example.com/1", "snippet": "相关内容摘要1"},
            {"title": "搜索结果2", "url": "https://example.com/2", "snippet": "相关内容摘要2"}
        ]
    }

        









