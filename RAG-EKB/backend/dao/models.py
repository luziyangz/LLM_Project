from sqlalchemy import Column, Integer, String, DateTime, Text, JSON, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

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
    meta_data = Column(JSON)  # 将 metadata 改为 meta_data

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
    meta_data = Column(JSON)  # 将 metadata 改为 meta_data