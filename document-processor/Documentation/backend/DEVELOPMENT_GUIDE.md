# 👨‍💻 Development Guide

Comprehensive guide for developers working on the Document Processing & AI Chat System.

## 📋 Table of Contents

- [Developer Onboarding](#developer-onboarding)
- [Development Environment Setup](#development-environment-setup)
- [Code Organization](#code-organization)
- [Development Workflows](#development-workflows)
- [Adding New Features](#adding-new-features)
- [Testing Strategies](#testing-strategies)
- [Code Style & Standards](#code-style--standards)
- [Debugging & Troubleshooting](#debugging--troubleshooting)
- [Performance Optimization](#performance-optimization)
- [Security Guidelines](#security-guidelines)

---

## 🚀 Developer Onboarding

### Prerequisites Knowledge

**Required Skills:**
- **Python 3.11+**: Advanced knowledge of Python and async programming
- **FastAPI**: RESTful API development and async endpoints
- **PostgreSQL**: Database design and SQL optimization
- **Vector Databases**: Understanding of embeddings and similarity search
- **Docker**: Containerization and deployment

**Recommended Skills:**
- **Machine Learning**: Basic understanding of transformers and embeddings
- **Supabase**: PostgreSQL with pgvector
- **React/Flutter**: For frontend integration testing
- **Redis**: Caching and session management

### Quick Start for New Developers

```bash
# 1. Clone and setup environment
git clone <repository-url>
cd Backend/document-processor

# 2. Create development environment
conda create -n py311 python=3.11
conda activate py311
pip install -r requirements.txt

# 3. Setup development tools
pip install pytest pytest-asyncio black flake8 mypy pre-commit

# 4. Install pre-commit hooks
pre-commit install

# 5. Copy environment template
cp .env.example .env
# Edit .env with your development credentials

# 6. Setup database
python -c "from database.database import create_tables; import asyncio; asyncio.run(create_tables())"

# 7. Run tests
pytest tests/

# 8. Start development server
python main.py
```

### Development Tools Setup

```bash
# Essential development tools
pip install -r requirements-dev.txt

# Contents of requirements-dev.txt:
# pytest==7.4.0
# pytest-asyncio==0.21.1
# pytest-mock==3.11.1
# black==23.7.0
# flake8==6.0.0
# mypy==1.5.1
# pre-commit==3.3.3
# httpx==0.24.1  # For testing API endpoints
# factory-boy==3.3.0  # For test data generation
```

---

## 🛠️ Development Environment Setup

### 1. **Local Development Configuration**

Create `.env.development`:

```env
# Development Environment
ENVIRONMENT=development
DEBUG=true

# Database
SUPABASE_URL=http://localhost:54321  # Local Supabase
SUPABASE_ANON_KEY=your_local_supabase_anon_key
SUPABASE_SERVICE_KEY=your_local_supabase_service_key
DATABASE_URL=sqlite:///./dev_database.db

# AI Services
HF_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2  # Faster for dev
HF_DEVICE=cpu  # Use CPU for development
EMBEDDING_CHUNK_SIZE=500  # Smaller chunks for faster processing
EMBEDDING_CHUNK_OVERLAP=100


# File Storage
UPLOAD_DIR=./storage/dev_uploads
MAX_FILE_SIZE_MB=10  # Smaller limit for development

# Logging
LOG_LEVEL=DEBUG
LOG_FILE=./logs/development.log
```

### 2. **IDE Configuration**

#### **VS Code Settings** (`.vscode/settings.json`)

```json
{
  "python.defaultInterpreterPath": "./venv/bin/python",
  "python.linting.enabled": true,
  "python.linting.flake8Enabled": true,
  "python.linting.mypyEnabled": true,
  "python.formatting.provider": "black",
  "python.formatting.blackArgs": ["--line-length=88"],
  "editor.formatOnSave": true,
  "editor.codeActionsOnSave": {
    "source.organizeImports": true
  },
  "files.exclude": {
    "**/__pycache__": true,
    "**/*.pyc": true,
    ".pytest_cache": true,
    ".coverage": true
  }
}
```

#### **PyCharm Configuration**

1. **Interpreter**: Set to conda environment (`py311`)
2. **Code Style**: Configure Black formatter
3. **Inspections**: Enable type checking and PEP 8
4. **Run Configurations**: Create configs for main.py and pytest

### 3. **Git Hooks Setup**

Create `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
      - id: check-merge-conflict

  - repo: https://github.com/psf/black
    rev: 23.7.0
    hooks:
      - id: black
        args: [--line-length=88]

  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
        args: [--max-line-length=88, --extend-ignore=E203,W503]

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.5.1
    hooks:
      - id: mypy
        additional_dependencies: [types-requests]
```

---

## 📂 Code Organization

### 1. **Project Structure Deep Dive**

```
Backend/document-processor/
├── 📁 api/                     # API Layer
│   ├── __init__.py
│   ├── endpoints.py            # All API routes
│   ├── dependencies.py         # FastAPI dependencies
│   └── middleware.py           # Custom middleware
│
├── 📁 services/                # Business Logic Layer
│   ├── __init__.py
│   ├── embedding_service.py    # Vector embeddings & search
│   ├── chatbot_service.py      # AI chat with context
│   ├── document_processor.py   # Document parsing & OCR
│   ├── file_handler.py         # File operations
│   ├── ocr_service.py          # OCR text extraction
│   ├── llm_service.py          # Large Language Model integration
│   ├── clip_service.py         # CLIP multimodal embeddings
│   └── technical_content_processor.py  # Technical documents
│
├── 📁 models/                  # Data Models
│   ├── __init__.py
│   ├── document.py             # Document-related schemas
│   ├── embedding.py            # Embedding & search schemas
│   └── response.py             # API response schemas
│
├── 📁 database/                # Data Access Layer
│   ├── __init__.py
│   ├── database.py             # Database connection & CRUD
│   ├── migrations/             # Database migrations
│   └── seeds/                  # Test data seeds
│
├── 📁 config/                  # Configuration
│   ├── __init__.py
│   ├── supabase_config.py      # Supabase client setup
│   ├── settings.py             # Application settings
│   └── logging_config.py       # Logging configuration
│
├── 📁 utils/                   # Utilities
│   ├── __init__.py
│   ├── text_processing.py      # Text manipulation utilities
│   ├── file_utils.py           # File handling utilities
│   ├── validation.py           # Input validation helpers
│   └── exceptions.py           # Custom exception classes
│
├── 📁 tests/                   # Test Suite
│   ├── __init__.py
│   ├── conftest.py             # Pytest configuration
│   ├── test_documents.py       # Document processing tests
│   ├── test_embeddings.py      # Embedding service tests
│   ├── test_chat.py            # Chatbot service tests
│   └── fixtures/               # Test data and fixtures
│
├── 📁 storage/                 # File Storage
│   ├── uploads/                # User uploads
│   ├── processed/              # Processed documents
│   ├── extracted_text/         # Extracted text files
│   ├── extracted_images/       # Extracted images
│   ├── thumbnails/             # Document thumbnails
│   └── temp/                   # Temporary processing files
│
├── 📁 logs/                    # Application Logs
├── 📁 docs/                    # Additional Documentation
├── .env                        # Environment variables
├── .env.example                # Environment template
├── requirements.txt            # Production dependencies
├── requirements-dev.txt        # Development dependencies
├── pytest.ini                 # Pytest configuration
├── mypy.ini                    # MyPy configuration
└── README.md                   # Main documentation
```

### 2. **Service Layer Design Patterns**

#### **Base Service Pattern**

```python
# services/base_service.py
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import logging

class BaseService(ABC):
    """Base class for all services with common functionality"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize service resources"""
        if not self._initialized:
            await self._setup()
            self._initialized = True
            self.logger.info(f"{self.__class__.__name__} initialized successfully")
    
    @abstractmethod
    async def _setup(self) -> None:
        """Service-specific initialization logic"""
        pass
    
    async def health_check(self) -> Dict[str, Any]:
        """Standard health check interface"""
        return {
            "service": self.__class__.__name__,
            "status": "healthy" if self._initialized else "not_initialized",
            "initialized": self._initialized
        }
    
    def __enter__(self):
        return self
    
    async def __aenter__(self):
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.cleanup()
    
    async def cleanup(self) -> None:
        """Cleanup service resources"""
        pass
```

#### **Service Implementation Example**

```python
# services/embedding_service.py
from .base_service import BaseService
from sentence_transformers import SentenceTransformer
from typing import List, Optional
import numpy as np

class EmbeddingService(BaseService):
    """Vector embedding generation and similarity search service"""
    
    def __init__(self):
        super().__init__()
        self.model: Optional[SentenceTransformer] = None
        self.model_name = "sentence-transformers/all-mpnet-base-v2"
        self.dimensions = 768
    
    async def _setup(self) -> None:
        """Initialize the embedding model"""
        try:
            self.model = SentenceTransformer(self.model_name)
            self.logger.info(f"Loaded embedding model: {self.model_name}")
        except Exception as e:
            self.logger.error(f"Failed to load embedding model: {e}")
            raise
    
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate vector embedding for text"""
        if not self._initialized:
            await self.initialize()
        
        try:
            embedding = self.model.encode(text, convert_to_tensor=False)
            return embedding.tolist()
        except Exception as e:
            self.logger.error(f"Failed to generate embedding: {e}")
            raise
    
    async def health_check(self) -> Dict[str, Any]:
        """Extended health check with model testing"""
        base_health = await super().health_check()
        
        if self._initialized:
            try:
                # Test embedding generation
                test_embedding = await self.generate_embedding("test")
                base_health.update({
                    "model_loaded": True,
                    "embedding_dimensions": len(test_embedding),
                    "test_successful": True
                })
            except Exception as e:
                base_health.update({
                    "model_loaded": False,
                    "error": str(e),
                    "test_successful": False
                })
        
        return base_health
```

### 3. **Error Handling Architecture**

```python
# utils/exceptions.py
class DocumentProcessorException(Exception):
    """Base exception for document processor"""
    pass

class DocumentProcessingError(DocumentProcessorException):
    """Document processing related errors"""
    pass

class EmbeddingServiceError(DocumentProcessorException):
    """Embedding service related errors"""
    pass

class ValidationError(DocumentProcessorException):
    """Input validation errors"""
    def __init__(self, message: str, field: str = None, value: Any = None):
        super().__init__(message)
        self.field = field
        self.value = value

# api/error_handlers.py
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
import logging

logger = logging.getLogger(__name__)

async def validation_exception_handler(request: Request, exc: ValidationError):
    """Handle validation errors"""
    logger.warning(f"Validation error: {exc}")
    return JSONResponse(
        status_code=400,
        content={
            "success": False,
            "error": {
                "code": "VALIDATION_ERROR",
                "message": str(exc),
                "field": exc.field,
                "value": exc.value
            }
        }
    )

```

---

## 🔄 Development Workflows

### 1. **Feature Development Workflow**

```bash
# 1. Create feature branch
git checkout -b feature/add-document-versioning

# 2. Set up development environment
conda activate py311
pip install -r requirements-dev.txt

# 3. Write failing tests first (TDD approach)
# Create tests/test_document_versioning.py

# 4. Implement feature
# Modify necessary services and models

# 5. Run tests continuously
pytest tests/ -v --tb=short

# 6. Check code quality
black .
flake8 .
mypy .

# 7. Update documentation
# Update API_REFERENCE.md if API changes
# Update ARCHITECTURE.md if design changes

# 8. Test integration
python -m pytest tests/integration/

# 9. Create pull request
git add .
git commit -m "feat: add document versioning with rollback support"
git push origin feature/add-document-versioning
```

### 2. **Testing Workflow**

```python
# tests/conftest.py
import pytest
import asyncio
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from main import app
from database.database import Base, get_db

# Test database
SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def db():
    """Create test database"""
    Base.metadata.create_all(bind=engine)
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=engine)

@pytest.fixture
def client(db):
    """Create test client"""
    def override_get_db():
        try:
            yield db
        finally:
            db.close()
    
    app.dependency_overrides[get_db] = override_get_db
    yield TestClient(app)
    app.dependency_overrides.clear()

```

### 3. **Database Migration Workflow**

```python
# database/migrations/migration_001_add_document_versions.py
from sqlalchemy import Column, String, DateTime, ForeignKey, Integer
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class DocumentVersion(Base):
    __tablename__ = "document_versions"
    
    id = Column(String, primary_key=True)
    document_id = Column(String, ForeignKey("documents_metadata.document_id"))
    version_number = Column(Integer, nullable=False)
    content_hash = Column(String, nullable=False)
    created_at = Column(DateTime, nullable=False)

# Migration script
def upgrade():
    """Add document versioning table"""
    from sqlalchemy import create_engine
    from database.database import DATABASE_URL
    
    engine = create_engine(DATABASE_URL)
    DocumentVersion.__table__.create(engine, checkfirst=True)

def downgrade():
    """Remove document versioning table"""
    from sqlalchemy import create_engine
    from database.database import DATABASE_URL
    
    engine = create_engine(DATABASE_URL)
    DocumentVersion.__table__.drop(engine, checkfirst=True)
```

---

## 🆕 Adding New Features

### 1. **Step-by-Step Feature Addition Guide**

#### **Example: Adding Document Collaboration Feature**

**Step 1: Define Requirements**
```markdown
# Feature: Document Collaboration
- Allow multiple users to collaborate on documents
- Track changes and provide version history
- Real-time notifications for collaborators
- Permission-based access control
```

**Step 2: Design API Endpoints**
```python
# Add to api/endpoints.py

@router.post("/document/{document_id}/collaborate")
async def invite_collaborator(
    document_id: str,
    collaborator_email: str = Form(...),
    permission_level: str = Form(...),  # read, write, admin
):
    """Invite user to collaborate on document"""
    pass

@router.get("/document/{document_id}/collaborators")
async def get_collaborators(
    document_id: str,
):
    """Get list of document collaborators"""
    pass

@router.put("/document/{document_id}/collaborator/{user_id}")
async def update_collaborator_permissions(
    document_id: str,
    user_id: str,
    permission_level: str = Form(...),
):
    """Update collaborator permissions"""
    pass
```

**Step 3: Create Data Models**
```python
# models/collaboration.py
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class CollaboratorInvite(BaseModel):
    document_id: str
    collaborator_email: str
    permission_level: str  # read, write, admin
    invited_by: str
    invited_at: datetime

class CollaboratorResponse(BaseModel):
    user_id: str
    email: str
    first_name: Optional[str]
    last_name: Optional[str]
    permission_level: str
    joined_at: datetime
    last_active: Optional[datetime]

class CollaborationStats(BaseModel):
    total_collaborators: int
    active_collaborators: int
    pending_invites: int
    recent_activity: List[dict]
```

**Step 4: Implement Service Logic**
```python
# services/collaboration_service.py
from .base_service import BaseService
from typing import List, Optional
from models.collaboration import CollaboratorInvite, CollaboratorResponse

class CollaborationService(BaseService):
    """Document collaboration management service"""
    
    async def _setup(self) -> None:
        """Initialize collaboration service"""
        self.supabase = get_supabase_client()
        self.notification_service = NotificationService()
    
    async def invite_collaborator(
        self, 
        document_id: str, 
        collaborator_email: str,
        permission_level: str,
        invited_by: str
    ) -> CollaboratorInvite:
        """Invite user to collaborate on document"""
        
        # Validate permission level
        if permission_level not in ["read", "write", "admin"]:
            raise ValidationError("Invalid permission level")
        
        # Check if user has permission to invite
        if not await self._can_invite(document_id, invited_by):
            raise ValidationError("Insufficient permissions to invite collaborators")
        
        # Create invitation
        invite = CollaboratorInvite(
            document_id=document_id,
            collaborator_email=collaborator_email,
            permission_level=permission_level,
            invited_by=invited_by,
            invited_at=datetime.utcnow()
        )
        
        # Store in database
        await self._store_invitation(invite)
        
        # Send notification
        await self.notification_service.send_collaboration_invite(invite)
        
        return invite
    
    async def _can_invite(self, document_id: str, user_id: str) -> bool:
        """Check if user can invite collaborators"""
        # Implementation depends on your permission system
        pass
```

**Step 5: Add Database Schema**
```sql
-- Add to supabase_schema.sql
CREATE TABLE document_collaborators (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID REFERENCES documents_metadata(document_id),
    user_id UUID REFERENCES users(id),
    permission_level VARCHAR(20) NOT NULL CHECK (permission_level IN ('read', 'write', 'admin')),
    invited_by UUID REFERENCES users(id),
    invited_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    accepted_at TIMESTAMP WITH TIME ZONE,
    last_active TIMESTAMP WITH TIME ZONE,
    status VARCHAR(20) DEFAULT 'pending' CHECK (status IN ('pending', 'accepted', 'rejected', 'removed'))
);

-- Add indexes for performance
CREATE INDEX idx_collaborators_document_id ON document_collaborators(document_id);
CREATE INDEX idx_collaborators_user_id ON document_collaborators(user_id);
CREATE INDEX idx_collaborators_status ON document_collaborators(status);

-- Add indexes for efficient querying
CREATE INDEX idx_collaborators_document_user ON document_collaborators(document_id, user_id);
```

**Step 6: Write Comprehensive Tests**
```python
# tests/test_collaboration.py
import pytest
from httpx import AsyncClient

class TestCollaboration:
    
    @pytest.mark.asyncio
    async def test_invite_collaborator_success(self, client, auth_headers, test_document):
        """Test successful collaborator invitation"""
        response = await client.post(
            f"/api/document/{test_document.document_id}/collaborate",
            data={
                "collaborator_email": "collaborator@example.com",
                "permission_level": "read"
            },
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["invitation"]["collaborator_email"] == "collaborator@example.com"
        assert data["invitation"]["permission_level"] == "read"
    
    @pytest.mark.asyncio
    async def test_invite_collaborator_invalid_permission(self, client, auth_headers, test_document):
        """Test invitation with invalid permission level"""
        response = await client.post(
            f"/api/document/{test_document.document_id}/collaborate",
            data={
                "collaborator_email": "collaborator@example.com",
                "permission_level": "invalid"
            },
            headers=auth_headers
        )
        
        assert response.status_code == 400
        data = response.json()
        assert data["success"] is False
        assert "Invalid permission level" in data["error"]["message"]
```

### 2. **API Endpoint Development Patterns**

#### **Standard Endpoint Structure**

```python
@router.post("/endpoint")
async def endpoint_name(
    # Path parameters
    document_id: str,
    
    # Query parameters
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0),
    
    # Request body
    request_data: RequestModel,
    
    # Form data (for file uploads)
    file: UploadFile = File(...),
    
    # Dependencies
    db: Session = Depends(get_db),
    
    # Background tasks
    background_tasks: BackgroundTasks = None
):
    """
    Endpoint description
    
    Args:
        document_id: Description of parameter
        request_data: Description of request body
        
    Returns:
        ResponseModel: Description of response
        
    Raises:
        HTTPException: Description of when this is raised
    """
    try:
        # 1. Validate input
        if not document_id:
            raise ValidationError("Document ID is required")
        
        # 2. Validate access
        if not await validate_document_access(document_id):
            raise ValidationError("Invalid document access")
        
        # 3. Business logic
        result = await service.process_request(request_data)
        
        # 4. Background tasks (if needed)
        if background_tasks:
            background_tasks.add_task(update_analytics, document_id)
        
        # 5. Return response
        return ResponseModel(
            success=True,
            data=result,
            message="Operation completed successfully"
        )
        
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in endpoint_name: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
```

---

## 🧪 Testing Strategies

### 1. **Test Pyramid Structure**

```
    ┌─────────────────┐
    │   E2E Tests     │  ← Few, expensive, full system tests
    │    (5-10%)      │
    ├─────────────────┤
    │ Integration     │  ← Medium number, test service interactions
    │   Tests (20%)   │
    ├─────────────────┤
    │   Unit Tests    │  ← Many, fast, test individual functions
    │    (70%)        │
    └─────────────────┘
```

### 2. **Unit Testing Patterns**

```python
# tests/test_embedding_service.py
import pytest
from unittest.mock import Mock, patch
from services.embedding_service import EmbeddingService

class TestEmbeddingService:
    
    @pytest.fixture
    def embedding_service(self):
        """Create embedding service instance for testing"""
        service = EmbeddingService()
        service.model = Mock()
        service._initialized = True
        return service
    
    @pytest.mark.asyncio
    async def test_generate_embedding_success(self, embedding_service):
        """Test successful embedding generation"""
        # Arrange
        test_text = "This is a test sentence"
        mock_embedding = [0.1, 0.2, 0.3]
        embedding_service.model.encode.return_value = mock_embedding
        
        # Act
        result = await embedding_service.generate_embedding(test_text)
        
        # Assert
        assert result == mock_embedding
        embedding_service.model.encode.assert_called_once_with(
            test_text, 
            convert_to_tensor=False
        )
    
    @pytest.mark.asyncio
    async def test_generate_embedding_not_initialized(self):
        """Test embedding generation when service not initialized"""
        # Arrange
        service = EmbeddingService()
        
        # Act & Assert
        with pytest.raises(RuntimeError, match="Service not initialized"):
            await service.generate_embedding("test")
    
    @pytest.mark.asyncio
    @patch('services.embedding_service.SentenceTransformer')
    async def test_service_initialization(self, mock_transformer):
        """Test service initialization"""
        # Arrange
        mock_model = Mock()
        mock_transformer.return_value = mock_model
        
        # Act
        service = EmbeddingService()
        await service.initialize()
        
        # Assert
        assert service._initialized is True
        assert service.model == mock_model
        mock_transformer.assert_called_once_with(service.model_name)
```

### 3. **Integration Testing**

```python
# tests/integration/test_document_processing_flow.py
import pytest
from httpx import AsyncClient
import tempfile
import os

class TestDocumentProcessingFlow:
    
    @pytest.mark.asyncio
    async def test_complete_document_processing_flow(self, client):
        """Test complete flow from upload to search"""
        
        # Step 1: Upload document
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("This is a test document about machine learning algorithms.")
            temp_file_path = f.name
        
        try:
            with open(temp_file_path, 'rb') as f:
                response = await client.post(
                    "/api/process-with-embeddings",
                    files={"file": ("test.txt", f, "text/plain")},
                    data={"chunk_size": "500", "chunk_overlap": "100"}
                )
            
            assert response.status_code == 200
            upload_data = response.json()
            document_id = upload_data["document_id"]
            
            # Step 2: Wait for processing (in real scenario, poll status)
            # For test, assume immediate processing
            
            # Step 3: Search for content
            search_response = await client.post(
                "/api/search-documents",
                json={
                    "query": "machine learning",
                    "limit": 5,
                    "similarity_threshold": 0.1
                }
            )
            
            assert search_response.status_code == 200
            search_data = search_response.json()
            assert search_data["success"] is True
            assert len(search_data["results"]) > 0
            assert search_data["results"][0]["document_id"] == document_id
            
            # Step 4: Test chat context preparation
            chat_response = await client.post(
                "/api/chat/prepare-context",
                json={
                    "message": "What is machine learning?",
                    "use_context": True,
                    "max_context_chunks": 3
                }
            )
            
            assert chat_response.status_code == 200
            chat_data = chat_response.json()
            assert chat_data["success"] is True
            assert "machine learning" in chat_data["formatted_prompt"].lower()
            
        finally:
            # Cleanup
            os.unlink(temp_file_path)
```

### 4. **Performance Testing**

```python
# tests/performance/test_embedding_performance.py
import pytest
import time
import asyncio
from services.embedding_service import EmbeddingService

class TestEmbeddingPerformance:
    
    @pytest.mark.asyncio
    async def test_embedding_generation_performance(self):
        """Test embedding generation performance benchmarks"""
        service = EmbeddingService()
        await service.initialize()
        
        # Test single embedding
        start_time = time.time()
        await service.generate_embedding("This is a test sentence for performance testing.")
        single_time = time.time() - start_time
        
        # Single embedding should complete in under 1 second
        assert single_time < 1.0
        
        # Test batch processing
        texts = [f"Test sentence number {i}" for i in range(100)]
        start_time = time.time()
        
        tasks = [service.generate_embedding(text) for text in texts]
        await asyncio.gather(*tasks)
        
        batch_time = time.time() - start_time
        avg_time_per_embedding = batch_time / len(texts)
        
        # Average time per embedding in batch should be much lower
        assert avg_time_per_embedding < 0.1
        
        print(f"Single embedding time: {single_time:.4f}s")
        print(f"Batch processing time: {batch_time:.4f}s")
        print(f"Average time per embedding: {avg_time_per_embedding:.4f}s")
```

---

## 🎨 Code Style & Standards

### 1. **Python Style Guidelines**

#### **Code Formatting**
```python
# Use Black formatter with 88 character line length
# Example of well-formatted code

from typing import List, Optional, Dict, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """
    Document processing service with OCR capabilities.
    
    This service handles document upload, text extraction, and metadata generation
    for various file formats including PDF, DOCX, XLSX, and images.
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.supported_formats = [".pdf", ".docx", ".xlsx", ".png", ".jpg"]
        self._initialized = False
    
    async def process_document(
        self,
        file_path: str,
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process document and extract text content.
        
        Args:
            file_path: Path to the document file
            options: Processing options (optional)
            
        Returns:
            Dict containing extracted text, metadata, and processing info
            
        Raises:
            DocumentProcessingError: If processing fails
            ValidationError: If file format is not supported
        """
        if not self._initialized:
            await self.initialize()
        
        # Validate file format
        if not self._is_supported_format(file_path):
            raise ValidationError(f"Unsupported file format: {file_path}")
        
        try:
            # Extract text based on file type
            extracted_text = await self._extract_text(file_path)
            metadata = await self._extract_metadata(file_path)
            
            return {
                "text": extracted_text,
                "metadata": metadata,
                "processed_at": datetime.utcnow().isoformat(),
                "file_path": file_path
            }
            
        except Exception as e:
            logger.error(f"Failed to process document {file_path}: {e}")
            raise DocumentProcessingError(f"Processing failed: {e}")
```

#### **Type Hints**
```python
# Always use type hints for better code clarity and IDE support

from typing import List, Dict, Optional, Union, Tuple, Any, AsyncGenerator
from datetime import datetime

# Function signatures
async def search_documents(
    query: str,
    user_id: str,
    limit: int = 10,
    similarity_threshold: float = 0.3
) -> List[Dict[str, Any]]:
    """Search documents with type-safe parameters"""
    pass

# Class attributes
class ChatService:
    model: Optional[SentenceTransformer]
    config: Dict[str, Any]
    initialized: bool = False
    
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.model = None

# Complex types
SearchResult = Dict[str, Union[str, float, int, datetime]]
ProcessingOptions = Dict[str, Union[str, int, float, bool]]
EmbeddingVector = List[float]
```

### 2. **Documentation Standards**

#### **Docstring Format**
```python
def complex_function(
    document_id: str,
    processing_options: Dict[str, Any],
    user_context: Optional[Dict[str, str]] = None
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Process document with advanced options and user context.
    
    This function handles complex document processing with user-specific
    customizations and returns detailed processing results.
    
    Args:
        document_id: Unique identifier for the document to process
        processing_options: Configuration options for processing:
            - chunk_size (int): Size of text chunks for embedding
            - chunk_overlap (int): Overlap between chunks
            - ocr_enabled (bool): Whether to enable OCR for images
            - language (str): Document language code (default: 'en')
        user_context: Optional user-specific context:
            - user_id (str): User identifier
            - preferences (str): User processing preferences
    
    Returns:
        Tuple containing:
            - success (bool): Whether processing succeeded
            - message (str): Success/error message
            - results (dict): Processing results with keys:
                - extracted_text (str): Extracted document text
                - metadata (dict): Document metadata
                - chunks (list): Text chunks for embedding
                - processing_time (float): Time taken in seconds
    
    Raises:
        DocumentProcessingError: When document processing fails
        ValidationError: When input parameters are invalid
        AuthenticationError: When user lacks permissions
    
    Example:
        >>> success, msg, results = await complex_function(
        ...     "doc-123",
        ...     {"chunk_size": 1000, "ocr_enabled": True},
        ...     {"user_id": "user-456"}
        ... )
        >>> if success:
        ...     print(f"Processed {len(results['chunks'])} chunks")
    
    Note:
        This function requires the embedding service to be initialized
        before calling. Processing time varies based on document size
        and OCR requirements.
    """
    pass
```

### 3. **Error Handling Patterns**

```python
# Structured error handling with proper logging

import logging
from typing import Any, Dict, Optional
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

class ServiceManager:
    """Example of comprehensive error handling"""
    
    async def process_with_error_handling(
        self, 
        document_id: str,
        user_id: str
    ) -> Dict[str, Any]:
        """Process document with comprehensive error handling"""
        
        processing_context = {
            "document_id": document_id,
            "user_id": user_id,
            "operation": "document_processing",
            "timestamp": datetime.utcnow().isoformat()
        }
        
        try:
            # Log operation start
            logger.info("Starting document processing", extra=processing_context)
            
            # Validate inputs
            await self._validate_inputs(document_id, user_id)
            
            # Perform processing with timeout
            async with self._processing_timeout(timeout=300):
                result = await self._process_document(document_id)
            
            # Log success
            logger.info("Document processing completed successfully", 
                       extra={**processing_context, "result_size": len(result)})
            
            return {
                "success": True,
                "data": result,
                "metadata": processing_context
            }
            
        except ValidationError as e:
            logger.warning("Validation failed", 
                          extra={**processing_context, "error": str(e)})
            return self._error_response("VALIDATION_ERROR", str(e), 400)
            
            
        except TimeoutError as e:
            logger.error("Processing timeout",
                        extra={**processing_context, "error": str(e)})
            return self._error_response("TIMEOUT_ERROR", "Processing timeout", 408)
            
        except Exception as e:
            logger.error("Unexpected error in document processing",
                        extra={**processing_context, "error": str(e)},
                        exc_info=True)
            return self._error_response("INTERNAL_ERROR", "Internal server error", 500)
    
    @asynccontextmanager
    async def _processing_timeout(self, timeout: int):
        """Context manager for processing timeout"""
        try:
            async with asyncio.timeout(timeout):
                yield
        except asyncio.TimeoutError:
            raise TimeoutError(f"Operation timed out after {timeout} seconds")
    
    def _error_response(self, code: str, message: str, status: int) -> Dict[str, Any]:
        """Standardized error response format"""
        return {
            "success": False,
            "error": {
                "code": code,
                "message": message,
                "status": status,
                "timestamp": datetime.utcnow().isoformat()
            }
        }
```

---

## 🐛 Debugging & Troubleshooting

### 1. **Logging Configuration**

```python
# config/logging_config.py
import logging
import logging.config
import os
from datetime import datetime

def setup_logging():
    """Configure comprehensive logging for the application"""
    
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "detailed": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s - %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S"
            },
            "simple": {
                "format": "%(levelname)s - %(name)s - %(message)s"
            },
            "json": {
                "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
                "format": "%(asctime)s %(name)s %(levelname)s %(filename)s %(lineno)d %(funcName)s %(message)s"
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": "INFO",
                "formatter": "simple",
                "stream": "ext://sys.stdout"
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "DEBUG",
                "formatter": "detailed",
                "filename": f"{log_dir}/application.log",
                "maxBytes": 10485760,  # 10MB
                "backupCount": 5
            },
            "error_file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "ERROR",
                "formatter": "json",
                "filename": f"{log_dir}/errors.log",
                "maxBytes": 10485760,
                "backupCount": 10
            }
        },
        "loggers": {
            "": {  # Root logger
                "level": "INFO",
                "handlers": ["console", "file", "error_file"]
            },
            "services": {
                "level": "DEBUG",
                "handlers": ["file"],
                "propagate": False
            },
            "api": {
                "level": "INFO",
                "handlers": ["console", "file"],
                "propagate": False
            }
        }
    }
    
    logging.config.dictConfig(logging_config)

# Usage in services
class EmbeddingService:
    def __init__(self):
        self.logger = logging.getLogger(f"services.{self.__class__.__name__}")
    
    async def process_embedding(self, text: str) -> List[float]:
        self.logger.debug(f"Processing embedding for text length: {len(text)}")
        
        try:
            start_time = time.time()
            embedding = self.model.encode(text)
            processing_time = time.time() - start_time
            
            self.logger.info(f"Embedding generated successfully", extra={
                "text_length": len(text),
                "embedding_dimensions": len(embedding),
                "processing_time": processing_time
            })
            
            return embedding.tolist()
            
        except Exception as e:
            self.logger.error(f"Embedding generation failed", extra={
                "text_length": len(text),
                "error": str(e),
                "error_type": type(e).__name__
            }, exc_info=True)
            raise
```

### 2. **Debugging Tools & Techniques**

```python
# utils/debug_tools.py
import functools
import time
import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)

def debug_performance(func: Callable) -> Callable:
    """Decorator to measure and log function performance"""
    
    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs) -> Any:
        start_time = time.time()
        
        try:
            result = await func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            logger.debug(f"{func.__name__} completed in {execution_time:.4f}s", extra={
                "function": func.__name__,
                "execution_time": execution_time,
                "args_count": len(args),
                "kwargs_count": len(kwargs)
            })
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"{func.__name__} failed after {execution_time:.4f}s", extra={
                "function": func.__name__,
                "execution_time": execution_time,
                "error": str(e),
                "error_type": type(e).__name__
            })
            raise
    
    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs) -> Any:
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            logger.debug(f"{func.__name__} completed in {execution_time:.4f}s")
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"{func.__name__} failed after {execution_time:.4f}s: {e}")
            raise
    
    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

# Usage example
class DocumentProcessor:
    
    @debug_performance
    async def extract_text(self, file_path: str) -> str:
        """Extract text with performance monitoring"""
        # Implementation here
        pass

# Memory debugging
import tracemalloc
import gc

class MemoryDebugger:
    """Memory usage debugging utilities"""
    
    @staticmethod
    def start_tracing():
        """Start memory tracing"""
        tracemalloc.start()
    
    @staticmethod
    def get_memory_snapshot():
        """Get current memory snapshot"""
        if not tracemalloc.is_tracing():
            return None
        
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')
        
        return {
            "top_memory_consumers": [
                {
                    "filename": stat.traceback.format()[0],
                    "size_mb": stat.size / 1024 / 1024,
                    "count": stat.count
                }
                for stat in top_stats[:10]
            ],
            "total_size_mb": sum(stat.size for stat in top_stats) / 1024 / 1024
        }
    
    @staticmethod
    def force_garbage_collection():
        """Force garbage collection and return collected objects"""
        collected = gc.collect()
        logger.info(f"Garbage collection completed, collected {collected} objects")
        return collected
```

### 3. **Common Issues & Solutions**

```python
# utils/troubleshooting.py
class TroubleshootingGuide:
    """Common issues and their solutions"""
    
    COMMON_ISSUES = {
        "embedding_service_initialization_failed": {
            "description": "EmbeddingService fails to initialize",
            "possible_causes": [
                "CUDA not available but HF_DEVICE=cuda",
                "Insufficient memory for model loading",
                "Network issues downloading model",
                "Incorrect model name in configuration"
            ],
            "solutions": [
                "Set HF_DEVICE=cpu in environment variables",
                "Increase available memory or use smaller model",
                "Check internet connection and HuggingFace Hub access",
                "Verify model name in HF_MODEL_NAME environment variable"
            ],
            "debug_commands": [
                "python -c \"import torch; print(torch.cuda.is_available())\"",
                "python -c \"from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')\"",
                "pip list | grep torch",
                "nvidia-smi"  # For CUDA debugging
            ]
        },
        
        "supabase_connection_failed": {
            "description": "Cannot connect to Supabase database",
            "possible_causes": [
                "Incorrect SUPABASE_URL or SUPABASE_ANON_KEY",
                "Network connectivity issues",
                "Supabase project paused or deleted",
                "Database schema not created"
            ],
            "solutions": [
                "Verify environment variables in .env file",
                "Test network connectivity to Supabase",
                "Check Supabase dashboard for project status",
                "Run supabase_schema.sql in Supabase SQL editor"
            ],
            "debug_commands": [
                "python -c \"from config.supabase_config import get_supabase_client; client = get_supabase_client(); print(client.table('users').select('*').limit(1).execute())\"",
                "curl -I $SUPABASE_URL",
                "nslookup your-project.supabase.co"
            ]
        },
        
        "ocr_processing_failed": {
            "description": "OCR text extraction fails",
            "possible_causes": [
                "Tesseract not installed or not in PATH",
                "Unsupported image format",
                "Corrupted image file",
                "Insufficient permissions to read file"
            ],
            "solutions": [
                "Install Tesseract OCR system dependency",
                "Convert image to supported format (PNG, JPG)",
                "Verify image file integrity",
                "Check file permissions and accessibility"
            ],
            "debug_commands": [
                "which tesseract",
                "tesseract --version",
                "python -c \"import pytesseract; print(pytesseract.get_tesseract_version())\"",
                "file /path/to/image.png"
            ]
        }
    }
    
    @classmethod
    def diagnose_issue(cls, issue_key: str) -> Dict[str, Any]:
        """Get diagnostic information for a specific issue"""
        if issue_key not in cls.COMMON_ISSUES:
            return {"error": f"Unknown issue: {issue_key}"}
        
        return cls.COMMON_ISSUES[issue_key]
    
    @classmethod
    def run_diagnostic_commands(cls, issue_key: str) -> Dict[str, str]:
        """Run diagnostic commands and return results"""
        issue = cls.COMMON_ISSUES.get(issue_key)
        if not issue:
            return {"error": f"Unknown issue: {issue_key}"}
        
        results = {}
        for cmd in issue.get("debug_commands", []):
            try:
                import subprocess
                result = subprocess.run(
                    cmd, 
                    shell=True, 
                    capture_output=True, 
                    text=True, 
                    timeout=30
                )
                results[cmd] = {
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "returncode": result.returncode
                }
            except Exception as e:
                results[cmd] = {"error": str(e)}
        
        return results
```

---

**This comprehensive development guide provides everything needed for developers to effectively work on the document processing system. For deployment information, see the [Deployment Guide](./DEPLOYMENT_GUIDE.md).**