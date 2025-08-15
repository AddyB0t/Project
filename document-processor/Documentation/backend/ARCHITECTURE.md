# 🏗️ System Architecture

Comprehensive architectural documentation for the Document Processing & AI Chat System.

## 📋 Table of Contents

- [System Overview](#system-overview)
- [Architecture Patterns](#architecture-patterns)
- [Core Components](#core-components)
- [Data Flow Diagrams](#data-flow-diagrams)
- [Database Architecture](#database-architecture)
- [Security Architecture](#security-architecture)
- [Integration Patterns](#integration-patterns)
- [Performance Considerations](#performance-considerations)
- [Scalability Design](#scalability-design)

---

## 🌐 System Overview

The Document Processing & AI Chat System is a **sophisticated, multi-layered architecture** designed for:

- **Document Processing**: Multi-format parsing, OCR, and text extraction
- **AI-Powered Search**: Vector embeddings and semantic similarity
- **Context-Aware Chat**: Intelligent responses using document context
- **Multimodal Processing**: Text and image understanding via CLIP

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Frontend Layer                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │   Flutter   │  │    React    │  │      Web Client       │ │
│  │   Mobile    │  │  Dashboard  │  │    (Any HTTP Client)   │ │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘ │
└─────────────────┬───────────────────────────────┬─────────────┘
                  │                               │
                  ▼                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                      API Gateway Layer                         │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                    FastAPI Server                          ││
│  │  • Request Validation & Rate Limiting                      ││
│  │  • CORS & Security Headers                                 ││
│  │  • Interactive Documentation (/docs)                       ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────┬───────────────────────────────┬─────────────┘
                  │                               │
                  ▼                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Service Layer                              │
│ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌───────────┐ │
│ │  Document   │ │  Embedding  │ │  Chatbot    │ │    File   │ │
│ │  Processor  │ │   Service   │ │  Service    │ │  Handler  │ │
│ └─────────────┘ └─────────────┘ └─────────────┘ └───────────┘ │
│ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌───────────┐ │
│ │    OCR      │ │    File     │ │    CLIP     │ │    LLM    │ │
│ │  Service    │ │  Handler    │ │  Service    │ │  Service  │ │
│ └─────────────┘ └─────────────┘ └─────────────┘ └───────────┘ │
└─────────────────┬───────────────────────────────┬─────────────┘
                  │                               │
                  ▼                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Data & Storage Layer                        │
│ ┌─────────────────────────┐  ┌─────────────────────────────────┐│
│ │     Supabase Database   │  │        Local Storage           ││
│ │  • PostgreSQL + pgvector│  │  • File System Storage        ││
│ │  • Document Metadata    │  │  • Document Uploads            ││
│ │  • Vector Embeddings    │  │  • Processed Files             ││
│ │  • Vector Embeddings    │  │  • Extracted Text/Images       ││
│ │  • Search Analytics     │  │  • Thumbnails & Previews       ││
│ │  • Row Level Security   │  │  • Temporary Processing        ││
│ └─────────────────────────┘  └─────────────────────────────────┘│
└─────────────────┬───────────────────────────────┬─────────────┘
                  │                               │
                  ▼                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                   External Services                            │
│ ┌─────────────────┐ ┌─────────────────┐ ┌───────────────────┐ │
│ │  HuggingFace    │ │     OpenAI      │ │     Tesseract     │ │
│ │  Transformers   │ │   (Optional)    │ │   OCR Engine      │ │
│ │  • Sentence     │ │  • GPT Models   │ │  • Text Extract   │ │
│ │    Transformers │ │  • Embeddings   │ │  • Multi-language │ │
│ │  • CLIP Model   │ │  • Chat Completion │ │  • Image OCR    │ │
│ └─────────────────┘ └─────────────────┘ └───────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔧 Architecture Patterns

### 1. **Layered Architecture**

The system follows a **clean, layered architecture** with clear separation of concerns:

```
┌─────────────────┐
│  Presentation   │  ← FastAPI endpoints, request/response handling
├─────────────────┤
│    Business     │  ← Services, business logic, workflows
├─────────────────┤
│   Data Access   │  ← Database operations, external API calls
├─────────────────┤
│   Persistence   │  ← Supabase, file system, caching
└─────────────────┘
```

### 2. **Service-Oriented Architecture (SOA)**

Each major functionality is encapsulated in dedicated services:

- **DocumentProcessor**: File parsing and text extraction
- **EmbeddingService**: Vector generation and similarity search
- **ChatbotService**: Context-aware AI responses
- **OCRService**: Image text extraction
- **FileHandler**: File upload and storage management

### 3. **Repository Pattern**

Data access is abstracted through repository-like interfaces:

```python
# Example: Database operations abstracted through service layer
class EmbeddingService:
    def __init__(self):
        self.supabase = get_supabase_client()
        self.model = SentenceTransformer('all-mpnet-base-v2')
    
    async def search_similar_documents(self, query: str) -> List[SearchResult]:
        # Business logic + data access abstraction
        pass
```

### 4. **Dependency Injection**

Services are injected and configured at startup:

```python
# main.py - Service initialization and injection
@app.on_event("startup")
async def startup_event():
    global embedding_service
    embedding_service = EmbeddingService()
    
    # Inject into endpoints module
    import api.endpoints as endpoints_module
    endpoints_module.embedding_service = embedding_service
```

---

## 🧩 Core Components

### 1. **API Layer** (`api/endpoints.py`)

**Responsibilities:**
- HTTP request/response handling
- Input validation and sanitization
- Rate limiting and CORS
- Error handling and response formatting

**Key Features:**
- FastAPI automatic documentation
- Pydantic model validation
- Multipart file upload handling
- Background task processing

```python
@router.post("/process-with-embeddings")
async def process_with_embeddings(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    chunk_size: int = Form(1000),
    chunk_overlap: int = Form(200)
):
    # Request handling with validation
    pass
```

### 2. **Document Processing Pipeline** (`services/document_processor.py`)

**Responsibilities:**
- Multi-format document parsing
- Text extraction with fallback strategies
- Image extraction and processing
- Metadata extraction
- File format validation

**Processing Flow:**
```
Document Upload → Format Detection → Text Extraction → Image Extraction → Metadata Generation
     │                    │               │                │                    │
     ▼                    ▼               ▼                ▼                    ▼
File Validation    Format-Specific    OCR Fallback    Image Processing    Structured Output
                     Parsing         (if needed)      (CLIP Ready)
```

### 3. **Embedding Service** (`services/embedding_service.py`)

**Responsibilities:**
- Vector embedding generation
- Semantic similarity search
- Multimodal embedding (text + image)
- Batch processing for efficiency
- Vector database operations

**Architecture:**
```python
class EmbeddingService:
    def __init__(self):
        self.model = SentenceTransformer('all-mpnet-base-v2')  # 768 dimensions
        self.clip_service = CLIPService()  # For image embeddings
        self.supabase = get_supabase_client()
        
    async def create_embeddings(self, text: str, document_id: str):
        # Text chunking → Vector generation → Database storage
        pass
    
    async def search_similar_documents(self, query: str, threshold: float = 0.3):
        # Query embedding → Cosine similarity search → Ranked results
        pass
```

### 5. **Chatbot Service** (`services/chatbot_service.py`)

**Responsibilities:**
- Context-aware response generation
- Document similarity search (k=8)
- Conversation history management
- Technical content processing
- Prompt engineering and formatting

**Context Flow:**
```
User Query → Embedding → Similarity Search → Context Retrieval → Prompt Formatting → Response
     │           │              │                   │                │              │
     ▼           ▼              ▼                   ▼                ▼              ▼
 Query Analysis  Vector Gen  Top-K Results    Document Chunks   LLM Prompt    AI Response
```

### 6. **Database Layer** (`database/database.py`)

**Responsibilities:**
- SQLAlchemy ORM configuration
- Connection pool management
- Transaction handling
- Local database fallback
- CRUD operations

**Dual Database Architecture:**
```python
# Primary: Supabase (PostgreSQL + pgvector)
supabase = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

# Fallback: Local SQLite
engine = create_engine("sqlite:///./document_processor.db")
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
```

---

## 📊 Data Flow Diagrams

### 1. **Document Upload & Processing Flow**

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Client    │    │   FastAPI   │    │  Document   │    │  Embedding  │
│             │    │   Server    │    │  Processor  │    │   Service   │
└──────┬──────┘    └──────┬──────┘    └──────┬──────┘    └──────┬──────┘
       │                  │                  │                  │
       │ POST /process    │                  │                  │
       │ + file upload    │                  │                  │
       ├─────────────────►│                  │                  │
       │                  │                  │                  │
       │                  │ validate & save  │                  │
       │                  ├─────────────────►│                  │
       │                  │                  │                  │
       │                  │                  │ extract text     │
       │                  │                  │ + OCR fallback   │
       │                  │                  │ (background)     │
       │                  │                  ├─────────────┐    │
       │                  │                  │             │    │
       │                  │                  │◄────────────┘    │
       │                  │                  │                  │
       │                  │                  │ create embeddings│
       │                  │                  ├─────────────────►│
       │                  │                  │                  │
       │                  │                  │                  │ chunk text
       │                  │                  │                  │ generate vectors
       │                  │                  │                  │ store in Supabase
       │                  │                  │                  ├────────────┐
       │                  │                  │                  │            │
       │                  │                  │                  │◄───────────┘
       │                  │                  │                  │
       │                  │                  │◄─────────────────┤
       │                  │                  │                  │
       │                  │◄─────────────────┤                  │
       │                  │                  │                  │
       │◄─────────────────┤                  │                  │
       │ response with    │                  │                  │
       │ document_id &    │                  │                  │
       │ processing status│                  │                  │
```

### 2. **AI Chat Context Preparation Flow**

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Client    │    │  Chatbot    │    │  Embedding  │    │  Supabase   │
│             │    │  Service    │    │   Service   │    │  Database   │
└──────┬──────┘    └──────┬──────┘    └──────┬──────┘    └──────┬──────┘
       │                  │                  │                  │
       │ POST /chat/      │                  │                  │
       │ prepare-context  │                  │                  │
       ├─────────────────►│                  │                  │
       │ + user message   │                  │                  │
       │                  │                  │                  │
       │                  │ embed query      │                  │
       │                  ├─────────────────►│                  │
       │                  │                  │                  │
       │                  │                  │ similarity search│
       │                  │                  │ (k=8, threshold) │
       │                  │                  ├─────────────────►│
       │                  │                  │                  │
       │                  │                  │                  │ vector search
       │                  │                  │                  │ + user filter
       │                  │                  │                  │ return top chunks
       │                  │                  │                  ├──────────┐
       │                  │                  │                  │          │
       │                  │                  │                  │◄─────────┘
       │                  │                  │                  │
       │                  │                  │◄─────────────────┤
       │                  │                  │ ranked results   │
       │                  │◄─────────────────┤                  │
       │                  │ context chunks   │                  │
       │                  │                  │                  │
       │                  │ format prompt    │                  │
       │                  │ + build context  │                  │
       │                  │ + conversation   │                  │
       │                  ├─────────────┐    │                  │
       │                  │             │    │                  │
       │                  │◄────────────┘    │                  │
       │                  │                  │                  │
       │◄─────────────────┤                  │                  │
       │ formatted prompt │                  │                  │
       │ + context summary│                  │                  │
       │ + metadata       │                  │                  │
```

### 3. **User Authentication Flow**

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Client    │    │    Auth     │    │  Supabase   │    │   Local DB  │
│             │    │  Service    │    │  Database   │    │  (Fallback) │
└──────┬──────┘    └──────┬──────┘    └──────┬──────┘    └──────┬──────┘
       │                  │                  │                  │
       │ POST /auth/login │                  │                  │
       ├─────────────────►│                  │                  │
       │ email + password │                  │                  │
       │                  │                  │                  │
       │                  │ authenticate     │                  │
       │                  ├─────────────────►│                  │
       │                  │                  │                  │
       │                  │                  │ verify user      │
       │                  │                  │ + check password │
       │                  │                  ├─────────────┐    │
       │                  │                  │             │    │
       │                  │                  │◄────────────┘    │
       │                  │                  │                  │
       │                  │◄─────────────────┤                  │
       │                  │ success/failure  │                  │
       │                  │                  │                  │
       │                  │ fallback to local│                  │
       │                  │ (if Supabase     │                  │
       │                  │ fails)           │                  │
       │                  ├─────────────────────────────────────►│
       │                  │                  │                  │
       │                  │                  │                  │ verify user
       │                  │                  │                  │ locally
       │                  │                  │                  ├─────────┐
       │                  │                  │                  │         │
       │                  │                  │                  │◄────────┘
       │                  │                  │                  │
       │                  │◄─────────────────────────────────────┤
       │                  │ success/failure  │                  │
       │                  │                  │                  │
       │                  │ generate JWT     │                  │
       │                  │ token            │                  │
       │                  ├─────────────┐    │                  │
       │                  │             │    │                  │
       │                  │◄────────────┘    │                  │
       │                  │                  │                  │
       │◄─────────────────┤                  │                  │
       │ JWT token +      │                  │                  │
       │ user info        │                  │                  │
```

---

## 🗄️ Database Architecture

### 1. **Supabase Schema Design**

The system uses **PostgreSQL with pgvector extension** for vector operations:

```sql
-- Core Tables Structure
┌─────────────────────────────────────────────────────────────────┐
│                        Database Schema                         │
├─────────────────┬─────────────────┬─────────────────────────────┤
│     users       │  documents_     │     document_embeddings     │
│                 │   metadata      │                             │
├─────────────────┼─────────────────┼─────────────────────────────┤
│ • id (UUID)     │ • document_id   │ • id (UUID)                 │
│ • email         │ • user_id       │ • document_id               │
│ • password_hash │ • filename      │ • chunk_id                  │
│ • first_name    │ • file_type     │ • chunk_text                │
│ • last_name     │ • file_size     │ • embedding (vector[768])   │
│ • is_active     │ • total_chunks  │ • chunk_index               │
│ • created_at    │ • total_chars   │ • metadata (JSONB)          │
│ • updated_at    │ • status        │ • created_at                │
│                 │ • created_at    │                             │
└─────────────────┴─────────────────┴─────────────────────────────┘
                           │
                           ▼
            ┌─────────────────────────────┐
            │    similarity_searches      │
            ├─────────────────────────────┤
            │ • id (UUID)                 │
            │ • user_id                   │
            │ • query_text                │
            │ • query_embedding (vector)  │
            │ • results_found             │
            │ • search_timestamp          │
            │ • metadata (JSONB)          │
            └─────────────────────────────┘
```

### 2. **Vector Indexing Strategy**

```sql
-- IVFFlat Index for fast similarity search
CREATE INDEX idx_document_embeddings_vector 
ON document_embeddings 
USING ivfflat (embedding vector_cosine_ops) 
WITH (lists = 100);

-- Additional performance indexes
CREATE INDEX idx_embeddings_document_id ON document_embeddings(document_id);
CREATE INDEX idx_embeddings_document_id ON document_embeddings(document_id);
CREATE INDEX idx_searches_timestamp ON similarity_searches(search_timestamp);
```

### 3. **Custom Database Functions**

```sql
-- Similarity search with user isolation
CREATE OR REPLACE FUNCTION search_similar_embeddings(
  query_embedding vector(768),
  match_threshold float,
  match_count int
)
RETURNS TABLE (
  document_id uuid,
  chunk_text text,
  similarity float,
  metadata jsonb
)
LANGUAGE plpgsql
AS $$
BEGIN
  RETURN QUERY
  SELECT 
    de.document_id,
    de.chunk_text,
    1 - (de.embedding <=> query_embedding) as similarity,
    de.metadata
  FROM document_embeddings de
  WHERE 1 - (de.embedding <=> query_embedding) > match_threshold
  ORDER BY de.embedding <=> query_embedding
  LIMIT match_count;
END;
$$;
```

---

## 🔐 Security Architecture

### 1. **Data Protection Layers**

1. **Transport Security**: HTTPS/TLS encryption
2. **Input Validation**: Pydantic model validation
3. **File Storage**: Secure file directories
4. **Database**: Encrypted connections
5. **Rate Limiting**: Per-IP limits
6. **CORS**: Cross-origin request handling

---

## 🔗 Integration Patterns

### 1. **External Service Integration**

```python
# Service abstraction pattern
class ExternalServiceBase:
    async def health_check(self) -> bool:
        raise NotImplementedError
    
    async def process(self, data: Any) -> Any:
        raise NotImplementedError

class HuggingFaceService(ExternalServiceBase):
    def __init__(self):
        self.model = SentenceTransformer('all-mpnet-base-v2')
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    async def health_check(self) -> bool:
        try:
            test_embedding = self.model.encode("test")
            return len(test_embedding) == 768
        except Exception:
            return False

class CLIPService(ExternalServiceBase):
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
```

### 2. **Error Handling & Resilience**

```python
# Circuit breaker pattern for external services
class ServiceCircuitBreaker:
    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    async def call_service(self, service_func, *args, **kwargs):
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.timeout:
                self.state = "HALF_OPEN"
            else:
                raise ServiceUnavailableError("Circuit breaker is OPEN")
        
        try:
            result = await service_func(*args, **kwargs)
            self.reset()
            return result
        except Exception as e:
            self.record_failure()
            raise e
```

### 3. **Caching Strategy**

```python
# Multi-level caching
class CacheManager:
    def __init__(self):
        self.memory_cache = {}  # In-memory for frequently accessed data
        self.redis_cache = None  # Distributed cache (optional)
        self.database_cache = None  # Database query result cache
    
    async def get_embedding(self, text_hash: str) -> Optional[List[float]]:
        # Level 1: Memory cache
        if text_hash in self.memory_cache:
            return self.memory_cache[text_hash]
        
        # Level 2: Redis cache (if available)
        if self.redis_cache:
            cached = await self.redis_cache.get(f"embedding:{text_hash}")
            if cached:
                embedding = json.loads(cached)
                self.memory_cache[text_hash] = embedding
                return embedding
        
        # Level 3: Database lookup
        # (Embeddings are computed if not found)
        return None
```

---

## ⚡ Performance Considerations

### 1. **Vector Search Optimization**

```sql
-- Optimized similarity search with pre-filtering
EXPLAIN ANALYZE
SELECT 
    de.document_id,
    de.chunk_text,
    1 - (de.embedding <=> $1) as similarity
FROM document_embeddings de
JOIN documents_metadata dm ON de.document_id = dm.document_id
WHERE dm.user_id = $2
    AND de.embedding <=> $1 < 0.7  -- Pre-filter with distance threshold
ORDER BY de.embedding <=> $1
LIMIT 8;

-- Results in efficient index usage:
-- Index Scan using idx_document_embeddings_vector
-- Cost: 0.00..15.23 rows=8 width=36
```

### 2. **Batch Processing Strategy**

```python
# Efficient batch embedding generation
class BatchEmbeddingProcessor:
    def __init__(self, batch_size: int = 32):
        self.batch_size = batch_size
        self.model = SentenceTransformer('all-mpnet-base-v2')
    
    async def process_document_chunks(self, chunks: List[str]) -> List[List[float]]:
        embeddings = []
        
        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i:i + self.batch_size]
            batch_embeddings = self.model.encode(
                batch,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_tensor=False
            )
            embeddings.extend(batch_embeddings.tolist())
            
            # Allow other async operations
            await asyncio.sleep(0)
        
        return embeddings
```

### 3. **Memory Management**

```python
# Memory-efficient document processing
class MemoryEfficientProcessor:
    def __init__(self, max_memory_mb: int = 512):
        self.max_memory_mb = max_memory_mb
        self.current_memory_usage = 0
    
    async def process_large_document(self, file_path: str) -> Iterator[str]:
        """Stream processing for large documents"""
        with open(file_path, 'rb') as file:
            # Process in chunks to manage memory
            while True:
                chunk = file.read(1024 * 1024)  # 1MB chunks
                if not chunk:
                    break
                
                # Process chunk and yield results
                processed_text = await self.extract_text_from_chunk(chunk)
                yield processed_text
                
                # Memory management
                if self.current_memory_usage > self.max_memory_mb * 1024 * 1024:
                    gc.collect()
                    self.current_memory_usage = 0
```

---

## 📈 Scalability Design

### 1. **Horizontal Scaling Strategy**

```
┌─────────────────────────────────────────────────────────────────┐
│                    Scalable Architecture                       │
├─────────────────┬─────────────────┬─────────────────────────────┤
│  Load Balancer  │   API Servers   │       Background Jobs       │
├─────────────────┼─────────────────┼─────────────────────────────┤
│ • NGINX/HAProxy │ • Multiple      │ • Celery Workers            │
│ • SSL Termination│   FastAPI      │ • Redis Queue               │
│ • Health Checks │   instances     │ • Distributed Processing    │
│ • Rate Limiting │ • Auto-scaling  │ • Fault Tolerance           │
└─────────────────┴─────────────────┴─────────────────────────────┘
                           │
                           ▼
            ┌─────────────────────────────┐
            │     Shared Storage          │
            ├─────────────────────────────┤
            │ • Distributed File System   │
            │ • Database Connection Pool  │
            │ • Redis for Caching        │
            │ • Message Queue (Optional)  │
            └─────────────────────────────┘
```

### 2. **Database Scaling**

```python
# Connection pooling and read replicas
class ScalableDatabase:
    def __init__(self):
        # Write operations - Primary database
        self.write_engine = create_engine(
            DATABASE_URL,
            pool_size=20,
            max_overflow=40,
            pool_timeout=30,
            pool_recycle=3600
        )
        
        # Read operations - Read replicas
        self.read_engines = [
            create_engine(replica_url, pool_size=10)
            for replica_url in READ_REPLICA_URLS
        ]
        
        self.read_engine_index = 0
    
    def get_read_engine(self):
        """Round-robin load balancing for read operations"""
        engine = self.read_engines[self.read_engine_index]
        self.read_engine_index = (self.read_engine_index + 1) % len(self.read_engines)
        return engine
```

### 3. **Microservice Architecture (Future)**

```
┌─────────────────────────────────────────────────────────────────┐
│                  Microservice Evolution                        │
├─────────────────┬─────────────────┬─────────────────────────────┤
│  Auth Service   │ Document Service│     Search Service          │
├─────────────────┼─────────────────┼─────────────────────────────┤
│ • User mgmt     │ • File processing│ • Vector operations         │
│ • JWT tokens    │ • OCR pipeline  │ • Similarity search         │
│ • Permissions   │ • Metadata      │ • Query optimization        │
│ • Rate limiting │ • File storage  │ • Caching layer            │
└─────────────────┴─────────────────┴─────────────────────────────┘
                           │
                           ▼
            ┌─────────────────────────────┐
            │      API Gateway            │
            ├─────────────────────────────┤
            │ • Request routing           │
            │ • Service discovery         │
            │ • Circuit breakers          │
            │ • Monitoring & logging      │
            └─────────────────────────────┘
```

### 4. **Caching Architecture**

```python
# Multi-tier caching strategy
class CachingArchitecture:
    def __init__(self):
        self.l1_cache = {}  # In-memory (each instance)
        self.l2_cache = Redis()  # Shared cache (Redis)
        self.l3_cache = MemcachedClient()  # Distributed cache
    
    async def get_similar_documents(self, query_hash: str, user_id: str):
        # L1: Instance memory (fastest)
        cache_key = f"search:{user_id}:{query_hash}"
        
        if cache_key in self.l1_cache:
            return self.l1_cache[cache_key]
        
        # L2: Redis shared cache
        cached_result = await self.l2_cache.get(cache_key)
        if cached_result:
            result = json.loads(cached_result)
            self.l1_cache[cache_key] = result
            return result
        
        # L3: Compute and cache
        result = await self.compute_similarity_search(query_hash, user_id)
        
        # Cache at all levels
        self.l1_cache[cache_key] = result
        await self.l2_cache.setex(cache_key, 3600, json.dumps(result))
        
        return result
```

---

## 🔍 Monitoring & Observability

### 1. **Application Metrics**

```python
# Comprehensive monitoring
class MetricsCollector:
    def __init__(self):
        self.request_count = Counter('http_requests_total')
        self.request_duration = Histogram('http_request_duration_seconds')
        self.embedding_generation_time = Histogram('embedding_generation_seconds')
        self.search_performance = Histogram('search_duration_seconds')
        self.active_users = Gauge('active_users_total')
    
    @self.request_duration.time()
    async def track_request(self, endpoint: str, method: str):
        self.request_count.labels(endpoint=endpoint, method=method).inc()
```

### 2. **Health Check Architecture**

```python
# Comprehensive health monitoring
class HealthCheckManager:
    async def comprehensive_health_check(self):
        checks = {
            "database": await self.check_database_connectivity(),
            "embedding_service": await self.check_embedding_service(),
            "supabase": await self.check_supabase_connectivity(),
            "file_storage": await self.check_file_storage(),
            "external_apis": await self.check_external_apis()
        }
        
        overall_status = "healthy" if all(checks.values()) else "degraded"
        
        return {
            "status": overall_status,
            "checks": checks,
            "timestamp": datetime.utcnow().isoformat()
        }
```

---

**This architecture documentation provides a comprehensive overview of the system's design, patterns, and scalability considerations. For implementation details, see the [Development Guide](./DEVELOPMENT_GUIDE.md) and [Deployment Guide](./DEPLOYMENT_GUIDE.md).**