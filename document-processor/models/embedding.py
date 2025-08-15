from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid

class EmbeddingChunk(BaseModel):
    """Model for individual text chunk with embedding"""
    chunk_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    chunk_text: str
    chunk_index: int
    embedding: List[float]
    metadata: Dict[str, Any] = Field(default_factory=dict)

class DocumentEmbedding(BaseModel):
    """Model for document with embeddings"""
    document_id: str
    original_filename: str
    file_type: str
    file_size: int
    total_chunks: int
    total_characters: int
    chunks: List[EmbeddingChunk]
    processing_status: str = "completed"
    embedding_status: str = "completed"
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)

class EmbeddingRequest(BaseModel):
    """Request model for creating embeddings"""
    text: str
    chunk_size: int = 1000
    chunk_overlap: int = 200
    metadata: Dict[str, Any] = Field(default_factory=dict)

class EmbeddingResponse(BaseModel):
    """Response model for embedding operations"""
    success: bool
    document_id: str
    total_chunks: int
    embedding_dimensions: int
    processing_time: float
    message: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

class SimilaritySearchRequest(BaseModel):
    """Request model for similarity search"""
    query: str
    limit: int = Field(default=5, ge=1, le=50)
    similarity_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    filter_metadata: Dict[str, Any] = Field(default_factory=dict)

class SimilaritySearchResult(BaseModel):
    """Individual search result"""
    document_id: str
    chunk_id: str
    chunk_text: str
    chunk_index: int
    similarity_score: float
    original_filename: str
    file_type: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

class SimilaritySearchResponse(BaseModel):
    """Response model for similarity search"""
    success: bool
    query: str
    results: List[SimilaritySearchResult]
    total_results: int
    search_time: float
    similarity_threshold: float
    message: str

class DocumentEmbeddingInfo(BaseModel):
    """Model for document embedding information"""
    document_id: str
    original_filename: str
    file_type: str
    file_size: int
    total_chunks: int
    total_characters: int
    embedding_dimensions: int
    processing_status: str
    embedding_status: str
    created_at: datetime
    updated_at: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class EmbeddingStats(BaseModel):
    """Model for embedding statistics"""
    total_documents: int
    total_chunks: int
    total_characters: int
    average_chunks_per_document: float
    embedding_dimensions: int
    last_updated: datetime

class BulkEmbeddingRequest(BaseModel):
    """Request model for bulk embedding processing"""
    document_ids: List[str]
    chunk_size: int = 1000
    chunk_overlap: int = 200
    force_reprocess: bool = False

class BulkEmbeddingResponse(BaseModel):
    """Response model for bulk embedding processing"""
    success: bool
    processed_documents: int
    failed_documents: int
    total_chunks_created: int
    processing_time: float
    failed_document_ids: List[str] = Field(default_factory=list)
    message: str