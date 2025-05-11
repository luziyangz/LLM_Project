from fastapi import FastAPI, Body
from pydantic import BaseModel
import os
import json
from typing import List, Optional
from datetime import datetime
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from utils.logs_utils import LoggerConfig, log_decorator
from utils.embedding_utils import EmbeddingModelLoader
from utils.db_utils import DatabaseManager  # 添加这行导入
import os
import logging
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from dotenv import load_dotenv

# 创建日志实例
logger = LoggerConfig(tag='APP-backend').get_logger()

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

# 添加CORS中间件允许跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中应该设置为特定域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@log_decorator(level=logging.INFO)
async def init():  # 修改为异步函数
    logger.debug("开始初始化服务...")
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
        


        logger.info("初始化完成")
    except Exception as e:
        logger.error(f"初始化失败: {str(e)}", exc_info=True)
        raise

from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

@log_decorator(level=logging.INFO)
async def initdb():
    logger.debug("开始初始化数据库...")
    try:
        # 从环境变量获取数据库配置
        db_url = f"mysql+pymysql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
        
        # 创建数据库管理器实例
        db_manager = DatabaseManager(db_url)
        
        # 初始化数据库连接
        engine, SessionLocal = db_manager.init_db()
        
        # 从dao导入模型

        from dao.models import Base, ChatSession, ChatMessage  # 使用小写的dao
        
        # 检查并创建表
        db_manager.check_and_create_tables(Base)
        
        logger.info("数据库初始化完成")
    except Exception as e:
        logger.error(f"数据库初始化失败: {str(e)}", exc_info=True)
        raise

#创建一个加载文件函数
@log_decorator(level=logging.INFO)
async def load_documents():  # 修改为异步函数
    logger.debug("开始加载文档...")
    try:

        # 在这里添加加载文档的代码
        logger.info("文档加载完成")
    except Exception as e:
        logger.error(f"文档加载失败: {str(e)}", exc_info=True)
        raise

#处理前端界面的请求：当前端发送用户消息时
@app.get("/static/")
async def read_root():
    return {"Hello": "World"}

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
@app.get("/health")
def health_check():
    logger.info("健康检查请求")
    return {"status": "healthy"}

# @app.post("/api/chat")
# async def chat(request: ChatRequest):
#     try:
#         return {"response": f"您好，我收到了您的消息：{request.message}"}
#     except Exception as e:
#         logger.error(f"处理聊天请求失败: {str(e)}", exc_info=True)
#         return {"error": "服务器处理请求失败，请稍后重试"}

@app.get("api/chat/history")
async def get_chat_history():
    try:
        #连接数据库，获取聊天历史记录
        

        # 这里可以添加获取聊天历史记录的逻辑
        return {"history": []}
    except Exception as e:
        logger.error(f"获取聊天历史记录失败: {str(e)}", exc_info=True)
        return {"error": "服务器处理请求失败，请稍后重试"}

# 处理流式SSE响应   
@app.get("/api/stream")
async def stream_response(
    query: str,  # 用户查询内容
    session_id: str = None,  # 可选的会话ID
    web_search: bool = False  # 是否启用联网搜索
):
    try:
        logger.info(f"收到流式请求 - 查询: {query}, 会话ID: {session_id}, 联网搜索: {web_search}")

        if not query:
            raise HTTPException(status_code=400, detail="Missing query parameter")

        # 使用 StreamingResponse 来处理异步生成器
        return StreamingResponse(
            proccess_stream_response(query, session_id, web_search),
            media_type='text/event-stream'
        )
            
    except Exception as e:
        logger.error(f"处理流式请求失败: {str(e)}", exc_info=True)
        # 对于错误情况，返回一个包含错误信息的事件流
        async def error_stream():
            yield f"data: {json.dumps({'event': 'error', 'data': str(e)})}\n\n"
        return StreamingResponse(
            error_stream(),
            media_type='text/event-stream'
        )

async def proccess_stream_response(
    query: str,
    session_id: str = None,
    web_search: bool = False
):
    logger.info(f"query: {query}, session_id: {session_id}, web_search: {web_search}")
    
    # 使用上下文管理器处理数据库会话
    async with AsyncSessionLocal() as db:
        try:
            # 1. 会话管理
            chat_session = None
            try:
                if session_id and session_id.strip():
                    # 使用 select 语句优化查询
                    stmt = select(ChatSession).where(ChatSession.id == session_id)
                    result = await db.execute(stmt)
                    chat_session = result.scalar_one_or_none()
                
                if not chat_session:
                    # 创建新会话
                    chat_session = ChatSession(
                        created_at=datetime.utcnow(),
                        last_activity=datetime.utcnow()
                    )
                    db.add(chat_session)
                    await db.commit()
                    await db.refresh(chat_session)
                    session_id = chat_session.id
                    history = []
                    logger.info(f"创建新会话: {session_id}")
                else:
                    # 更新现有会话
                    chat_session.last_activity = datetime.utcnow()
                    # 获取历史对话，使用 select 优化查询
                    stmt = select(ChatMessage)\
                        .where(ChatMessage.session_id == session_id)\
                        .order_by(ChatMessage.timestamp.desc())\
                        .limit(5)
                    result = await db.execute(stmt)
                    history = result.scalars().all()[::-1]
                    logger.info(f"使用现有会话: {session_id}")
                
                await db.commit()
            except Exception as e:
                await db.rollback()
                logger.error(f"会话管理失败: {str(e)}", exc_info=True)
                yield {
                    "event": "error",
                    "data": json.dumps({"error": "会话管理失败，请重试"})
                }
                return

            # 2. 联网搜索处理
            search_context = ""
            if web_search:
                try:
                    search_results = await perform_web_search(query)
                    search_context = f"网络搜索结果:\n{search_results}"
                except Exception as e:
                    logger.error(f"联网搜索失败: {str(e)}")
                    search_context = "联网搜索暂时不可用"
                    # 不中断流程，继续执行

            # 3. 生成回答
            try:
                response_stream = generate_llm_response(
                    query=query,
                    history=history,
                    search_context=search_context,
                    session_id=session_id
                )

                # 4. 流式响应处理
                last_content = None  # 用于保存最后一条消息
                async for chunk in response_stream:
                    try:
                        if isinstance(chunk, dict) and chunk.get("error"):
                            yield {
                                "event": "error",
                                "data": json.dumps({"error": chunk["error"]})
                            }
                            continue

                        if chunk.get("content"):
                            last_content = chunk["content"]  # 保存最后一条消息
                            yield {
                                "event": "message",
                                "data": json.dumps({
                                    "type": "stream",
                                    "content": chunk["content"],
                                    "session_id": session_id
                                })
                            }

                            # 5. 如果是最后一条消息，保存对话记录
                            if chunk.get("is_final", False) and last_content:
                                try:
                                    await save_chat_messages(
                                        db=db,
                                        session_id=session_id,
                                        user_message=query,
                                        assistant_message=last_content
                                    )
                                except Exception as e:
                                    logger.error(f"保存对话记录失败: {str(e)}")
                                    # 继续处理流式响应，但记录错误

                    except Exception as e:
                        logger.error(f"处理单个响应块失败: {str(e)}", exc_info=True)
                        yield {
                            "event": "error",
                            "data": json.dumps({"error": "处理响应块失败"})
                        }

            except Exception as e:
                logger.error(f"生成回答失败: {str(e)}", exc_info=True)
                yield {
                    "event": "error",
                    "data": json.dumps({"error": "生成回答失败，请重试"})
                }

        except Exception as e:
            logger.error(f"处理流式响应失败: {str(e)}", exc_info=True)
            yield {
                "event": "error",
                "data": json.dumps({
                    "error": f"服务器处理请求失败: {str(e)}"
                })
            }

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

async def generate_llm_response(query, history, search_context, session_id):
    """生成LLM响应的异步生成器"""
    try:
        # 构建上下文
        context = []
        if search_context:
            context.append(search_context)
        
        if history:
            formatted_history = "\n".join([
                f"{'用户' if msg.role == 'user' else '助手'}: {msg.content}"
                for msg in history
            ])
            context.append(f"历史对话:\n{formatted_history}")

        # 构建提示词
        prompt = f"""
        基于以下信息回答用户问题:
        
        {' '.join(context)}
        
        用户问题: {query}
        
        请提供准确、相关的回答。如果使用了网络搜索信息，请注明来源。
        """

        # 调用模型生成回答
        async for chunk in dashscope.Generation.acall(
            model='qwen-max',
            prompt=prompt,
            stream=True,
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            result_format='message'
        ):
            if chunk.status_code == 200:
                content = chunk.output.choices[0].message.content
                yield {
                    "content": content,
                    "is_final": chunk.output.choices[0].finish_reason is not None
                }
            else:
                yield {"error": f"模型调用失败: {chunk.message}"}

    except Exception as e:
        logger.error(f"生成回答失败: {str(e)}")
        yield {"error": str(e)}

# 处理文件上传请求




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


        









