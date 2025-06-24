from sqlalchemy import Column, String, DateTime, Text, Integer, Float, JSON, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
import uuid

Base = declarative_base()

class ChatSession(Base):
    __tablename__ = "chat_sessions"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(50), index=True)
    title = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    status = Column(String(20), default="active")
    model_name = Column(String(50), nullable=True)
    system_prompt = Column(Text, nullable=True)
    total_tokens = Column(Integer, default=0)
    meta_data = Column(JSON, nullable=True)
    embedding = Column(Text, nullable=True)
    parent_message_id = Column(String(50), nullable=True)
    
    # 添加与 ChatMessage 的关系
    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan")

class ChatMessage(Base):
    __tablename__ = 'chat_messages'
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String(36), ForeignKey('chat_sessions.id'), nullable=False)
    role = Column(String(20), nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, nullable=False)
    
    # 添加与 ChatSession 的关系
    session = relationship("ChatSession", back_populates="messages")

class Document(Base):
    __tablename__ = "documents"
    
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False, unique=True)
    original_name = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)
    file_size = Column(Integer, nullable=False)
    file_type = Column(String(50), nullable=False)
    
    # 处理状态相关字段
    processing_status = Column(String(20), default="uploaded")  # uploaded, processing, processed, failed
    processed_at = Column(DateTime, nullable=True)
    processing_config = Column(Text, nullable=True)  # JSON格式存储处理配置
    
    # 处理结果相关字段
    chunk_count = Column(Integer, default=0)
    vector_dimension = Column(Integer, nullable=True)
    index_size = Column(Integer, default=0)
    processing_time = Column(Float, default=0.0)
    error_message = Column(Text, nullable=True)
    
    # 时间戳
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # 元数据
    word_count = Column(Integer, default=0)
    language = Column(String(10), nullable=True)
    encoding = Column(String(20), nullable=True)
