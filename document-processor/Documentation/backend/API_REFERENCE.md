# 📚 API Reference

Complete API documentation for the Document Processing & AI Chat System.

## 🔗 Base Information

- **Base URL**: `http://localhost:8000/api`
- **Content-Type**: `application/json` (unless specified otherwise)
- **Interactive Documentation**: `http://localhost:8000/docs`

## 📋 Table of Contents

- [Document Processing Endpoints](#document-processing-endpoints)
- [Search & Embedding Endpoints](#search--embedding-endpoints)
- [AI Chat Endpoints](#ai-chat-endpoints)
- [Admin & System Endpoints](#admin--system-endpoints)
- [Error Handling](#error-handling)
- [Rate Limiting](#rate-limiting)
- [Response Formats](#response-formats)

---


## 📄 Document Processing Endpoints

### 1. Upload Document

**`POST /api/upload-document`**

Upload and process a document (basic processing without embeddings).

#### Headers
```
Content-Type: multipart/form-data
```

#### Request Body (Form Data)
```
file: [document file] (PDF, DOCX, XLSX, PPTX, PNG, JPG, etc.)
```

#### Response
```json
{
  "success": true,
  "message": "Document uploaded and processed successfully",
  "document_id": "doc-uuid-456",
  "filename": "example.pdf",
  "file_type": "application/pdf",
  "file_size": 1024000,
  "processing_status": "completed",
  "extracted_text_preview": "This is the beginning of the extracted text...",
  "total_characters": 5420,
  "processed_at": "2024-01-15T10:35:00Z"
}
```

#### cURL Example
```bash
curl -X POST "http://localhost:8000/api/upload-document" \
  -F "file=@/path/to/your/document.pdf"
```

---

### 2. Process with Embeddings

**`POST /api/process-with-embeddings`**

Upload and process document with vector embedding generation.

#### Headers
```
Content-Type: multipart/form-data
```

#### Request Body (Form Data)
```
file: [document file]
chunk_size: 1000 (optional, default: 1000)
chunk_overlap: 200 (optional, default: 200)
```

#### Response
```json
{
  "success": true,
  "message": "Document processed with embeddings successfully",
  "document_id": "doc-uuid-789",
  "filename": "technical_manual.pdf",
  "file_type": "application/pdf",
  "file_size": 2048000,
  "processing_status": "completed",
  "embedding_status": "completed",
  "total_chunks": 25,
  "total_characters": 12850,
  "embedding_dimensions": 768,
  "processing_time": 45.2,
  "processed_at": "2024-01-15T10:40:00Z"
}
```

#### cURL Example
```bash
curl -X POST "http://localhost:8000/api/process-with-embeddings" \
  -F "file=@/path/to/document.pdf" \
  -F "chunk_size=1000" \
  -F "chunk_overlap=200"
```

---

### 3. Get Document Text

**`GET /api/document/{document_id}/text`**

Retrieve extracted text from a processed document.

#### Headers
```
# No special headers required
```

#### Path Parameters
- `document_id`: UUID of the document

#### Response
```json
{
  "success": true,
  "document_id": "doc-uuid-456",
  "filename": "example.pdf",
  "extracted_text": "Full extracted text content goes here...",
  "total_characters": 5420,
  "extraction_method": "direct",  // or "ocr"
  "processed_at": "2024-01-15T10:35:00Z"
}
```

#### cURL Example
```bash
curl -X GET "http://localhost:8000/api/document/doc-uuid-456/text"
```

---

### 4. Get User Documents

**`GET /api/documents`**

List all documents for the authenticated user.

#### Headers
```
# No special headers required
```

#### Query Parameters
- `limit`: Number of documents to return (default: 10, max: 100)
- `offset`: Number of documents to skip (default: 0)
- `status`: Filter by status (`processing`, `completed`, `failed`)

#### Response
```json
{
  "success": true,
  "documents": [
    {
      "document_id": "doc-uuid-456",
      "filename": "example.pdf",
      "file_type": "application/pdf",
      "file_size": 1024000,
      "processing_status": "completed",
      "embedding_status": "completed",
      "total_chunks": 15,
      "total_characters": 5420,
      "uploaded_at": "2024-01-15T10:35:00Z"
    }
  ],
  "total_count": 1,
  "limit": 10,
  "offset": 0
}
```

#### cURL Example
```bash
curl -X GET "http://localhost:8000/api/documents?limit=10&offset=0"
```

---

## 🔍 Search & Embedding Endpoints

### 1. Search Documents

**`POST /api/search-documents`**

Perform semantic search across user's documents.

#### Headers
```
Content-Type: application/json
```

#### Request Body
```json
{
  "query": "machine learning algorithms",
  "limit": 5,                    // Optional, default: 5, max: 20
  "similarity_threshold": 0.3,   // Optional, default: 0.3
  "document_types": ["pdf"],     // Optional, filter by file types
  "include_metadata": true       // Optional, default: false
}
```

#### Response
```json
{
  "success": true,
  "query": "machine learning algorithms",
  "results": [
    {
      "document_id": "doc-uuid-789",
      "chunk_id": "chunk-uuid-123",
      "chunk_text": "Machine learning algorithms are computational methods that enable...",
      "similarity_score": 0.89,
      "original_filename": "ml_guide.pdf",
      "chunk_index": 5,
      "file_type": "application/pdf",
      "metadata": {
        "page_number": 12,
        "section": "Introduction to ML"
      }
    }
  ],
  "total_results": 1,
  "search_time": 0.156,
  "similarity_threshold_used": 0.3
}
```

#### cURL Example
```bash
curl -X POST "http://localhost:8000/api/search-documents" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "machine learning algorithms",
    "limit": 5,
    "similarity_threshold": 0.3
  }'
```

---

### 2. Get Document Embeddings

**`GET /api/document/{document_id}/embeddings`**

Retrieve embeddings for a specific document.

#### Headers
```
# No special headers required
```

#### Path Parameters
- `document_id`: UUID of the document

#### Query Parameters
- `limit`: Number of chunks to return (default: 10)
- `offset`: Number of chunks to skip (default: 0)

#### Response
```json
{
  "success": true,
  "document_id": "doc-uuid-789",
  "filename": "technical_manual.pdf",
  "total_chunks": 25,
  "chunks": [
    {
      "chunk_id": "chunk-uuid-001",
      "chunk_index": 0,
      "chunk_text": "Introduction to the system architecture...",
      "embedding_dimensions": 768,
      "created_at": "2024-01-15T10:40:00Z"
    }
  ],
  "limit": 10,
  "offset": 0
}
```

#### cURL Example
```bash
curl -X GET "http://localhost:8000/api/document/doc-uuid-789/embeddings?limit=10"
```

---

### 3. Delete Document Embeddings

**`DELETE /api/document/{document_id}/embeddings`**

Delete all embeddings for a specific document.

#### Headers
```
# No special headers required
```

#### Path Parameters
- `document_id`: UUID of the document

#### Response
```json
{
  "success": true,
  "message": "Document embeddings deleted successfully",
  "document_id": "doc-uuid-789",
  "deleted_chunks": 25
}
```

#### cURL Example
```bash
curl -X DELETE "http://localhost:8000/api/document/doc-uuid-789/embeddings"
```

---

### 4. Get Embedding Statistics

**`GET /api/embeddings/stats`**

Get statistics about user's embeddings.

#### Headers
```
# No special headers required
```

#### Response
```json
{
  "success": true,
  "statistics": {
    "total_documents": 15,
    "total_chunks": 450,
    "total_characters": 125000,
    "average_chunks_per_document": 30,
    "embedding_model": "sentence-transformers/all-mpnet-base-v2",
    "embedding_dimensions": 768,
    "storage_size_mb": 12.5,
    "last_updated": "2024-01-15T10:45:00Z"
  }
}
```

#### cURL Example
```bash
curl -X GET "http://localhost:8000/api/embeddings/stats"
```

---

## 🤖 AI Chat Endpoints

### 1. Prepare Chat Context

**`POST /api/chat/prepare-context`**

Prepare AI chat context using k=8 similarity search.

#### Headers
```
Content-Type: application/json
```

#### Request Body
```json
{
  "message": "What is machine learning?",
  "use_context": true,                    // Optional, default: true
  "max_context_chunks": 8,               // Optional, default: 8
  "similarity_threshold": 0.3,           // Optional, default: 0.3
  "conversation_history": [              // Optional
    {
      "role": "user",
      "content": "Hello"
    },
    {
      "role": "assistant", 
      "content": "Hi! How can I help you?"
    }
  ],
  "include_technical_context": true      // Optional, for technical documents
}
```

#### Response
```json
{
  "success": true,
  "formatted_prompt": "You are an AI assistant with access to relevant document context...\n\nContext from documents:\n1. [ml_guide.pdf] Machine learning is a subset of artificial intelligence...",
  "context_summary": "Used context from 3 document chunks:\n- ml_guide.pdf: 2 chunks (avg similarity: 0.85)\n- ai_handbook.pdf: 1 chunk (similarity: 0.78)",
  "context_metadata": {
    "total_chunks": 3,
    "unique_documents": 2,
    "avg_similarity": 0.82,
    "max_similarity": 0.91,
    "min_similarity": 0.78
  },
  "search_time": 0.234,
  "context_chunks": [
    {
      "document_id": "doc-uuid-789",
      "chunk_text": "Machine learning is a subset of artificial intelligence...",
      "similarity_score": 0.91,
      "filename": "ml_guide.pdf",
      "chunk_index": 3
    }
  ],
  "message": "Prepared chat context with 3 relevant document chunks"
}
```

#### cURL Example
```bash
curl -X POST "http://localhost:8000/api/chat/prepare-context" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is machine learning?",
    "use_context": true,
    "max_context_chunks": 8,
    "similarity_threshold": 0.3
  }'
```

---

### 2. Search Chat Context

**`POST /api/chat/search-context`**

Search for relevant context without preparing full chat prompt.

#### Headers
```
# No special headers required
```

#### Query Parameters
- `query`: Search query (required)
- `max_chunks`: Maximum chunks to return (default: 8)
- `similarity_threshold`: Minimum similarity score (default: 0.3)

#### Response
```json
{
  "success": true,
  "query": "neural networks",
  "context_chunks": [
    {
      "document_id": "doc-uuid-789",
      "chunk_text": "Neural networks are computing systems inspired by biological neural networks...",
      "similarity_score": 0.95,
      "filename": "deep_learning.pdf",
      "chunk_index": 7
    }
  ],
  "total_chunks": 1,
  "search_time": 0.123
}
```

#### cURL Example
```bash
curl -X POST "http://localhost:8000/api/chat/search-context?query=neural%20networks&max_chunks=5"
```

---

### 3. Format Chat Prompt

**`POST /api/chat/format-prompt`**

Format a chat prompt with provided context chunks.

#### Headers
```
Content-Type: application/json
```

#### Request Body
```json
{
  "user_message": "Explain neural networks",
  "context_chunks": [
    {
      "chunk_text": "Neural networks are computing systems...",
      "filename": "deep_learning.pdf",
      "similarity_score": 0.95
    }
  ],
  "conversation_history": [],            // Optional
  "system_prompt": "You are an AI tutor"  // Optional
}
```

#### Response
```json
{
  "success": true,
  "formatted_prompt": "You are an AI tutor with access to relevant document context...\n\nUser: Explain neural networks",
  "context_summary": "Using 1 document chunk from deep_learning.pdf",
  "prompt_length": 1250
}
```

#### cURL Example
```bash
curl -X POST "http://localhost:8000/api/chat/format-prompt" \
  -H "Content-Type: application/json" \
  -d '{
    "user_message": "Explain neural networks",
    "context_chunks": [
      {
        "chunk_text": "Neural networks are computing systems...",
        "filename": "deep_learning.pdf",
        "similarity_score": 0.95
      }
    ]
  }'
```

---

### 4. Get Context Statistics

**`GET /api/chat/context-stats`**

Get statistics about chat context usage.

#### Headers
```
# No special headers required
```

#### Response
```json
{
  "success": true,
  "stats": {
    "total_searches": 150,
    "avg_chunks_per_search": 5.2,
    "avg_similarity_score": 0.67,
    "most_queried_topics": [
      {"topic": "machine learning", "count": 45},
      {"topic": "neural networks", "count": 32}
    ],
    "context_usage_last_30_days": 89,
    "last_search": "2024-01-15T10:45:00Z"
  }
}
```

#### cURL Example
```bash
curl -X GET "http://localhost:8000/api/chat/context-stats"
```

---

## ⚙️ Admin & System Endpoints

### 1. System Status

**`GET /api/status`**

Get comprehensive system status and health information.

#### Response
```json
{
  "server": "running",
  "embedding_service": "ready",
  "python_version": "3.11.0",
  "environment": "py311",
  "supabase": {
    "status": "connected",
    "embeddings_count": 1250,
    "error": null
  },
  "environment_variables": {
    "supabase_url_set": true,
    "supabase_key_set": true,
    "hf_model": "sentence-transformers/all-mpnet-base-v2"
  },
  "services": {
    "auth_service": "ready",
    "document_processor": "ready",
    "ocr_service": "ready",
    "llm_service": "ready"
  }
}
```

#### cURL Example
```bash
curl -X GET "http://localhost:8000/api/status"
```

---

### 2. Health Check

**`GET /health`**

Simple health check endpoint.

#### Response
```json
{
  "status": "healthy",
  "service": "document-processor",
  "version": "1.0.0"
}
```

#### cURL Example
```bash
curl -X GET "http://localhost:8000/health"
```

---

### 3. Service Validation

**`GET /api/validate-services`**

Validate all services and dependencies.

#### Headers
```
# No special headers required
```

#### Response
```json
{
  "validation_completed": true,
  "services_ready": true,
  "errors": [],
  "validations": {
    "embedding_service": {
      "status": "ready",
      "model_loaded": true,
      "test_embedding_successful": true
    },
    "supabase_connection": {
      "status": "connected",
      "pgvector_enabled": true,
      "tables_exist": true
    },
    "ocr_service": {
      "status": "ready",
      "tesseract_available": true
    }
  }
}
```

#### cURL Example
```bash
curl -X GET "http://localhost:8000/api/validate-services"
```

---

## ❌ Error Handling

### Standard Error Response Format

All endpoints return errors in the following format:

```json
{
  "success": false,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid email format",
    "details": {
      "field": "email",
      "provided_value": "invalid-email"
    }
  },
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### Common Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `VALIDATION_ERROR` | 400 | Invalid request data |
| `INSUFFICIENT_PERMISSIONS` | 403 | User lacks required permissions |
| `RESOURCE_NOT_FOUND` | 404 | Requested resource doesn't exist |
| `DUPLICATE_RESOURCE` | 409 | Resource already exists |
| `RATE_LIMIT_EXCEEDED` | 429 | Too many requests |
| `PROCESSING_ERROR` | 422 | Document processing failed |
| `SERVICE_UNAVAILABLE` | 503 | External service temporarily unavailable |
| `INTERNAL_ERROR` | 500 | Unexpected server error |


### Processing Errors

#### Unsupported File Type
```json
{
  "success": false,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Unsupported file type",
    "details": {
      "supported_types": ["pdf", "docx", "xlsx", "pptx", "png", "jpg", "jpeg"]
    }
  }
}
```

#### File Size Limit Exceeded
```json
{
  "success": false,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "File size exceeds maximum limit",
    "details": {
      "max_size_mb": 50,
      "provided_size_mb": 75
    }
  }
}
```

---

## 🚦 Rate Limiting

### Default Limits

- **Document processing**: 10 uploads per hour per user
- **Search endpoints**: 100 requests per minute per user
- **Chat endpoints**: 50 requests per minute per user

### Rate Limit Headers

All responses include rate limiting information:

```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1640995200
X-RateLimit-Type: user
```

### Rate Limit Exceeded Response

```json
{
  "success": false,
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "message": "Rate limit exceeded",
    "details": {
      "limit": 100,
      "reset_time": "2024-01-15T11:00:00Z"
    }
  }
}
```

---

## 📊 Response Formats

### Success Response

All successful responses follow this structure:

```json
{
  "success": true,
  "message": "Operation completed successfully",  // Optional
  "data": {
    // Response data here
  },
  "metadata": {  // Optional
    "timestamp": "2024-01-15T10:30:00Z",
    "processing_time": 0.234
  }
}
```

### Pagination

Endpoints that return lists include pagination information:

```json
{
  "success": true,
  "data": [...],
  "pagination": {
    "total_count": 150,
    "limit": 10,
    "offset": 20,
    "has_next": true,
    "has_previous": true
  }
}
```

---

## 🔧 Frontend Integration Examples

### Flutter/Dart Integration

```dart
class DocumentApiService {
  final String baseUrl;
  
  DocumentApiService({required this.baseUrl});
  
  // Upload and process document
  Future<Map<String, dynamic>> uploadDocument(File file) async {
    var request = http.MultipartRequest(
      'POST',
      Uri.parse('$baseUrl/api/process-with-embeddings'),
    );
    
    request.files.add(await http.MultipartFile.fromPath('file', file.path));
    
    var response = await request.send();
    var responseData = await response.stream.bytesToString();
    return jsonDecode(responseData);
  }
  
  // Search documents
  Future<Map<String, dynamic>> searchDocuments({
    required String query,
    int limit = 5,
    double similarityThreshold = 0.3,
  }) async {
    final response = await http.post(
      Uri.parse('$baseUrl/api/search-documents'),
      headers: {
        'Content-Type': 'application/json',
      },
      body: jsonEncode({
        'query': query,
        'limit': limit,
        'similarity_threshold': similarityThreshold,
      }),
    );
    
    return jsonDecode(response.body);
  }
  
  // Prepare chat context
  Future<Map<String, dynamic>> prepareChatContext({
    required String message,
    bool useContext = true,
    int maxContextChunks = 8,
    double similarityThreshold = 0.3,
  }) async {
    final response = await http.post(
      Uri.parse('$baseUrl/api/chat/prepare-context'),
      headers: {
        'Content-Type': 'application/json',
      },
      body: jsonEncode({
        'message': message,
        'use_context': useContext,
        'max_context_chunks': maxContextChunks,
        'similarity_threshold': similarityThreshold,
      }),
    );
    
    return jsonDecode(response.body);
  }
}
```

### JavaScript/React Integration

```javascript
class DocumentAPI {
  constructor(baseURL) {
    this.baseURL = baseURL;
  }
  
  async uploadDocument(file, options = {}) {
    const formData = new FormData();
    formData.append('file', file);
    
    if (options.chunkSize) formData.append('chunk_size', options.chunkSize);
    if (options.chunkOverlap) formData.append('chunk_overlap', options.chunkOverlap);
    
    const response = await fetch(`${this.baseURL}/api/process-with-embeddings`, {
      method: 'POST',
      body: formData,
    });
    
    return response.json();
  }
  
  async searchDocuments(query, options = {}) {
    const response = await fetch(`${this.baseURL}/api/search-documents`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        query,
        limit: options.limit || 5,
        similarity_threshold: options.similarityThreshold || 0.3,
      }),
    });
    
    return response.json();
  }
  
  async prepareChatContext(message, options = {}) {
    const response = await fetch(`${this.baseURL}/api/chat/prepare-context`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        message,
        use_context: options.useContext !== false,
        max_context_chunks: options.maxContextChunks || 8,
        similarity_threshold: options.similarityThreshold || 0.3,
        conversation_history: options.conversationHistory || [],
      }),
    });
    
    return response.json();
  }
}
```

---

## 📝 Notes

- All timestamps are in ISO 8601 format with UTC timezone
- File uploads have a maximum size limit of 50MB
- Vector embeddings use 768 dimensions with sentence-transformers
- Similarity search uses cosine similarity with configurable thresholds
- The system supports multimodal processing for images using CLIP
- User data is isolated using Row Level Security (RLS) policies

---

**For more information, see:**
- [System Architecture](./ARCHITECTURE.md)
- [Development Guide](./DEVELOPMENT_GUIDE.md)
- [Deployment Guide](./DEPLOYMENT_GUIDE.md)