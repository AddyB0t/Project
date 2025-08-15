from pydantic import BaseModel
from typing import Optional, Any, Dict

class APIResponse(BaseModel):
    """Generic API response model"""
    success: bool
    message: str
    data: Optional[Any] = None
    error: Optional[str] = None

class UploadProgressResponse(BaseModel):
    """Upload progress response model"""
    document_id: str
    progress: float  # 0-100
    status: str
    message: str

class ProcessingStatusResponse(BaseModel):
    """Document processing status response"""
    document_id: str
    status: str  # pending, processing, completed, failed
    progress: float
    message: str
    extracted_text: Optional[str] = None
    error: Optional[str] = None

class ValidationErrorResponse(BaseModel):
    """Validation error response model"""
    error: str = "validation_error"
    message: str
    details: Dict[str, Any]

class ServiceHealthResponse(BaseModel):
    """Service health check response"""
    status: str
    service: str
    version: str
    uptime: Optional[float] = None
    dependencies: Optional[Dict[str, str]] = None