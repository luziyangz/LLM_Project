# RAG-EKB 数据库设计与使用指南

## 一、数据库设计思路

### 1. 数据库选型
- 选择 MySQL 作为关系型数据库，用于存储结构化数据
- 选择 Faiss 作为向量数据库，用于存储和检索向量化的文本数据

### 2. 表结构设计

#### 2.1 会话表 (chat_sessions)
```sql
CREATE TABLE chat_sessions (
    id INT PRIMARY KEY AUTO_INCREMENT,  -- 自增主键
    session_id VARCHAR(50) UNIQUE,      -- 会话唯一标识
    user_id VARCHAR(50),                -- 用户标识
    title VARCHAR(255),                 -- 会话标题
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,  -- 创建时间
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,  -- 更新时间
    status VARCHAR(20) DEFAULT 'active',  -- 会话状态
    model_name VARCHAR(50),             -- 使用的模型
    system_prompt TEXT,                 -- 系统提示词
    total_tokens INT DEFAULT 0,         -- token总数
    metadata JSON                       -- 元数据
);
```

**字段说明**：
- `id`：自增主键，用于唯一标识记录
- `session_id`：会话唯一标识，用于前端引用
- `user_id`：用户标识，关联用户信息
- `title`：会话标题，显示在会话列表中
- `created_at`：创建时间，自动设置为当前时间
- `updated_at`：更新时间，自动更新为最新操作时间
- `status`：会话状态，如"active"、"archived"等
- `model_name`：使用的模型名称
- `system_prompt`：系统提示词，设定会话基调
- `total_tokens`：会话消耗的总token数
- `metadata`：元数据，存储额外信息的JSON字段

#### 2.2 消息表 (chat_messages)
```sql
CREATE TABLE chat_messages (
    id INT PRIMARY KEY AUTO_INCREMENT,  -- 自增主键
    session_id VARCHAR(50),             -- 关联的会话ID
    role VARCHAR(20),                   -- 角色(user/assistant)
    content TEXT,                       -- 消息内容
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,  -- 创建时间
    tokens INT,                         -- token数量
    embedding TEXT,                     -- 向量嵌入
    parent_message_id VARCHAR(50),      -- 父消息ID
    status VARCHAR(20) DEFAULT 'success',  -- 消息状态
    metadata JSON,                      -- 元数据
    FOREIGN KEY (session_id) REFERENCES chat_sessions(session_id)  -- 外键约束
);
```

**字段说明**：
- `id`：自增主键，唯一标识消息
- `session_id`：关联的会话ID，外键关联chat_sessions表
- `role`：消息角色，如"user"、"assistant"、"system"
- `content`：消息文本内容
- `created_at`：消息创建时间
- `tokens`：消息使用的token数量
- `embedding`：消息的向量嵌入表示
- `parent_message_id`：父消息ID，用于构建消息树
- `status`：消息状态，如"success"、"failed"等
- `metadata`：元数据，存储额外信息

## 二、数据库环境配置

### 1. MySQL安装与配置
1. 下载并安装MySQL：https://dev.mysql.com/downloads/installer/
2. 安装过程中设置root密码
3. 确保MySQL服务已启动：
   ```bash
   # 检查MySQL服务状态
   net start | findstr "MySQL"
   
   # 如未启动，可手动启动
   net start MySQL80
   ```

### 2. 环境变量配置
在项目根目录创建`.env`文件，配置数据库连接信息：
```
# 数据库配置
DB_USER=rag_user
DB_PASSWORD=your_password
DB_HOST=localhost
DB_PORT=3306
DB_NAME=rag_ekb
```

### 3. 安装Python依赖
确保已安装所有必要的Python包：
```bash
pip install -r requirements.txt
```

## 三、Navicat操作教程

### 1. 创建数据库连接
1. 打开Navicat Premium
2. 点击左上角"连接" → 选择"MySQL"
3. 填写连接信息：
   - 连接名：RAG-EKB（自定义名称）
   - 主机：localhost
   - 端口：3306
   - 用户名：root
   - 密码：您的MySQL root密码
4. 点击"测试连接"确保连接成功
5. 点击"确定"保存连接

### 2. 创建数据库
1. 在左侧连接上右键点击
2. 选择"新建数据库"
3. 填写数据库信息：
   - 数据库名：rag_ekb
   - 字符集：utf8mb4（支持完整Unicode字符集）
   - 排序规则：utf8mb4_unicode_ci（不区分大小写的Unicode排序）
4. 点击"确定"创建数据库

### 3. 创建专用数据库用户
1. 在左侧连接上右键点击
2. 选择"用户和权限"
3. 点击"新建用户"按钮
4. 填写用户信息：
   - 用户名：rag_user
   - 主机：localhost
   - 密码：设置安全密码（与.env文件中一致）
5. 切换到"服务器权限"标签
6. 找到rag_ekb数据库，勾选所需权限：
   - SELECT（查询）
   - INSERT（插入）
   - UPDATE（更新）
   - DELETE（删除）
   - CREATE（创建表）
   - DROP（删除表）
7. 点击"确定"保存用户设置

### 4. 创建数据表
1. 在左侧数据库上右键点击
2. 选择"新建表"
3. 设计表结构：
   - 添加字段名称、类型、长度等
   - 设置主键、索引、默认值等
   - 设置外键关系
4. 点击"保存"创建表

## 四、代码实现说明

### 1. 数据模型定义（DAO/models.py）
```python
from sqlalchemy import Column, Integer, String, DateTime, Text, JSON, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

# 创建基础模型类
Base = declarative_base()

# 会话表定义
class ChatSession(Base):
    __tablename__ = "chat_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(50), unique=True, index=True)
    user_id = Column(String(50), index=True)
    title = Column(String(255))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    status = Column(String(20), default="active")
    model_name = Column(String(50))
    system_prompt = Column(Text)
    total_tokens = Column(Integer, default=0)
    metadata = Column(JSON)

# 消息表定义
class ChatMessage(Base):
    __tablename__ = "chat_messages"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(50), ForeignKey("chat_sessions.session_id"), index=True)
    role = Column(String(20))
    content = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    tokens = Column(Integer)
    embedding = Column(Text, nullable=True)
    parent_message_id = Column(String(50), nullable=True)
    status = Column(String(20), default="success")
    metadata = Column(JSON)
```

### 2. 数据库工具类（utils/db_utils.py）
```python
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker
from utils.logs_utils import LoggerConfig, log_decorator
import logging

logger = LoggerConfig().get_logger()

class DatabaseManager:
    def __init__(self, connection_string):
        self.connection_string = connection_string
        self.engine = None
        self.SessionLocal = None
        
    @log_decorator(level=logging.INFO)
    def init_db(self):
        try:
            # 创建数据库引擎，配置连接池
            self.engine = create_engine(
                self.connection_string,
                pool_size=5,  # 连接池大小
                max_overflow=10,  # 最大溢出连接数
                pool_timeout=30  # 连接超时时间
            )
            
            # 创建会话工厂
            self.SessionLocal = sessionmaker(
                autocommit=False,  # 关闭自动提交
                autoflush=False,  # 关闭自动刷新
                bind=self.engine  # 绑定引擎
            )
            
            return self.engine, self.SessionLocal
            
        except Exception as e:
            logger.error(f"数据库初始化失败: {str(e)}", exc_info=True)
            raise
            
    def check_and_create_tables(self, Base):
        try:
            # 获取数据库检查器
            inspector = inspect(self.engine)
            # 获取现有表列表
            existing_tables = inspector.get_table_names()
            
            # 检查每个表是否存在，不存在则创建
            for table_name in Base.metadata.tables.keys():
                if table_name not in existing_tables:
                    Base.metadata.tables[table_name].create(bind=self.engine)
                    logger.info(f"创建表 {table_name} 成功")
                else:
                    logger.info(f"表 {table_name} 已存在，跳过创建")
                    
        except Exception as e:
            logger.error(f"检查和创建表失败: {str(e)}", exc_info=True)
            raise
```

### 3. 数据库初始化（app.py）
```python
from dotenv import load_dotenv
import os

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
        
        # 从DAO导入模型
        from DAO.models import Base, ChatSession, ChatMessage
        
        # 检查并创建表
        db_manager.check_and_create_tables(Base)
        
        logger.info("数据库初始化完成")
    except Exception as e:
        logger.error(f"数据库初始化失败: {str(e)}", exc_info=True)
        raise
```

## 五、维护建议

### 1. 数据库安全
- 定期更改数据库密码
- 限制数据库访问IP
- 避免使用root账户直接连接
- 定期备份重要数据

### 2. 性能优化
- 为常用查询字段建立索引
- 定期优化表结构
- 监控慢查询并优化
- 合理设置连接池参数

### 3. 开发规范
- 使用ORM避免SQL注入
- 事务管理确保数据一致性
- 异常处理和日志记录
- 定期清理无用数据

## 六、常见问题解决

### 1. 连接问题
- 检查MySQL服务是否运行
- 验证用户名密码是否正确
- 确认防火墙是否允许3306端口
- 检查主机名解析是否正确

### 2. 权限问题
- 确认用户是否有足够权限
- 检查表权限设置
- 验证数据库访问权限

### 3. 性能问题
- 检查慢查询日志
- 优化索引使用
- 监控连接池状态
- 分析查询执行计划
