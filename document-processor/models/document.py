from sqlalchemy import Column, String, Integer, DateTime, Text, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from pydantic import BaseModel
from typing import Optional
from datetime import datetime

Base = declarative_base()

class Document(Base):
    __tablename__ = "documents"
    
    id = Column(String, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    original_filename = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    pdf_path = Column(String)
    file_size = Column(Integer, nullable=False)
    file_type = Column(String, nullable=False)
    extracted_text = Column(Text)
    processing_status = Column(String, default="pending")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    processing_time = Column(Float)
    user_id = Column(String, nullable=True, index=True)

class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    original_filename: str
    file_size: int
    file_type: str
    processing_status: str
    extracted_text: Optional[str] = None
    created_at: datetime
    processing_time: Optional[float] = None
    user_id: Optional[str] = None

class DocumentTextResponse(BaseModel):
    document_id: str
    filename: str
    extracted_text: str
    processing_status: str
    created_at: datetime
    processing_time: Optional[float] = None

class DocumentListResponse(BaseModel):
    documents: list[DocumentUploadResponse]
    total: int

class ErrorResponse(BaseModel):
    error: str
    message: str
    details: Optional[str] = None