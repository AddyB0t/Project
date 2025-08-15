# 📄 Document Processing & AI Chat System

[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-green.svg)](https://fastapi.tiangolo.com/)
[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![Supabase](https://img.shields.io/badge/Supabase-Vector%20DB-orange.svg)](https://supabase.com)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)](https://huggingface.co)

A **sophisticated document processing and conversational AI system** with multimodal capabilities, semantic search, and context-aware chatbot responses. Built with FastAPI, Supabase vector database, and HuggingFace transformers.

## 🌟 **Key Features**

### 📚 **Document Processing**
- **Multi-format support**: PDF, DOCX, XLSX, PPTX, images (PNG, JPG, etc.)
- **Advanced OCR**: PyMuPDF + Tesseract with intelligent preprocessing
- **Smart text extraction**: Direct text extraction with OCR fallback
- **Batch processing**: Efficient handling of multiple documents
- **Technical content processing**: Specialized handling for engineering/architectural documents

### 🧠 **AI & Machine Learning**
- **Vector embeddings**: 768-dimension embeddings using sentence-transformers
- **Multimodal processing**: CLIP integration for text + image embeddings
- **Semantic similarity search**: Configurable thresholds and k-nearest neighbor
- **Context-aware chatbot**: k=8 similarity search for intelligent responses
- **Technical content analysis**: Specialized processing for complex documents

### 🗄️ **Database & Storage**
- **Supabase integration**: PostgreSQL with pgvector extension
- **Vector similarity search**: Custom functions with cosine similarity
- **Efficient indexing**: IVFFlat indexing for fast vector operations
- **Comprehensive analytics**: Search tracking and usage statistics

## 🚀 **Quick Start**

### Prerequisites
- **Python 3.11+**
- **Conda** (recommended) or virtual environment
- **Supabase account** and project
- **Tesseract OCR** (system dependency)

### 1. Environment Setup

```bash
# Create conda environment
conda create -n py311 python=3.11
conda activate py311

# Navigate to project
cd Backend/document-processor

# Install dependencies
pip install -r requirements.txt
```

### 2. System Dependencies

#### Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr tesseract-ocr-eng poppler-utils
```

#### macOS
```bash
brew install tesseract poppler
```

#### Windows
1. Download Tesseract from [GitHub releases](https://github.com/UB-Mannheim/tesseract/wiki)
2. Add Tesseract to your PATH
3. Install poppler: `conda install poppler`

### 3. Configuration

Create `.env` file in the project root:

```env
# Supabase Configuration
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=your_supabase_anon_key
SUPABASE_SERVICE_KEY=your_supabase_service_key

# AI Model Configuration
HF_MODEL_NAME=sentence-transformers/all-mpnet-base-v2
HF_DEVICE=cuda  # or 'cpu' if no GPU available

# Processing Configuration
EMBEDDING_CHUNK_SIZE=1000
EMBEDDING_CHUNK_OVERLAP=200
EMBEDDING_DIMENSIONS=768

# Processing Configuration (continued)
# Add other configuration as needed
```

### 4. Database Setup

1. **Enable pgvector extension** in Supabase:
   - Go to Database → Extensions
   - Enable the `vector` extension

2. **Run database schema**:
   ```bash
   # Copy and paste supabase_schema.sql into Supabase SQL editor
   cat supabase_schema.sql
   ```

### 5. Start the Server

```bash
python main.py
```

The API will be available at `http://localhost:8000`

**Interactive API documentation**: `http://localhost:8000/docs`

## 📋 **API Overview**

| Category | Endpoint | Description |
|----------|----------|-------------|
| **Documents** | `POST /api/upload-document` | Upload and process document |
| | `POST /api/process-with-embeddings` | Process with embeddings |
| | `GET /api/document/{id}/text` | Get extracted text |
| **Search** | `POST /api/search-documents` | Semantic document search |
| | `GET /api/document/{id}/embeddings` | Get document embeddings |
| **AI Chat** | `POST /api/chat/prepare-context` | Prepare chat with context |
| | `POST /api/chat/search-context` | Search chat context |
| | `POST /api/chat/format-prompt` | Format AI prompt |
| **Analytics** | `GET /api/embeddings/stats` | Embedding statistics |
| | `GET /api/status` | System status |

> **Detailed API documentation**: See [API_REFERENCE.md](./API_REFERENCE.md)

## 🤖 **AI Chat Integration**

### Context-Aware Responses

The system uses **k=8 similarity search** to provide intelligent, context-aware responses:

1. **User Query** → **Similarity Search** → **Context Retrieval**
2. **Document Chunks** → **Formatted Prompt** → **AI Response**

### Frontend Integration Example

```dart
// Flutter integration example
Future<Map<String, dynamic>> getChatContext(String userMessage) async {
  final response = await http.post(
    Uri.parse('$backendUrl/api/chat/prepare-context'),
    headers: {'Content-Type': 'application/json'},
    body: jsonEncode({
      'message': userMessage,
      'use_context': true,
      'max_context_chunks': 8,
      'similarity_threshold': 0.3
    }),
  );
  
  return jsonDecode(response.body);
}
```

### Configuration Options

- **max_context_chunks**: Number of relevant chunks to retrieve (default: 8)
- **similarity_threshold**: Minimum similarity score (default: 0.3)
- **use_context**: Enable/disable document context (default: true)

## 🏗️ **System Architecture**

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Frontend      │    │   FastAPI        │    │   Supabase      │
│   (Flutter/Web) │◄──►│   Backend        │◄──►│   PostgreSQL    │
│                 │    │                  │    │   + pgvector    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │  HuggingFace     │
                    │  Transformers    │
                    │  + CLIP Model    │
                    └──────────────────┘
```

### Core Components

- **Document Processor**: Multi-format parsing and OCR
- **Embedding Service**: Vector generation and similarity search
- **Chat Service**: Context-aware AI responses
- **Database Layer**: Supabase with vector extensions

> **Detailed architecture**: See [ARCHITECTURE.md](./ARCHITECTURE.md)

## 📁 **Project Structure**

```
Backend/document-processor/
├── 📄 README.md                 # This documentation
├── 🚀 main.py                   # FastAPI application entry
├── 📦 requirements.txt          # Python dependencies
├── 🗄️ supabase_schema.sql       # Database schema
├── ⚙️ .env                      # Environment variables (create this)
│
├── 🔌 api/                      # API endpoints
│   └── endpoints.py             # All routes and handlers
│
├── 🛠️ services/                 # Business logic
│   ├── embedding_service.py     # Vector embeddings & search
│   ├── chatbot_service.py       # AI chat with context
│   ├── document_processor.py    # Document processing & OCR
│   ├── file_handler.py          # File upload & management
│   ├── ocr_service.py           # OCR text extraction
│   ├── llm_service.py           # LLM integration
│   ├── clip_service.py          # Multimodal embeddings
│   └── technical_content_processor.py  # Technical documents
│
├── 📊 models/                   # Data models
│   ├── document.py              # Document-related models
│   ├── embedding.py             # Embedding & search models
│   └── response.py              # API response models
│
├── ⚙️ config/                   # Configuration
│   └── supabase_config.py       # Supabase client setup
│
├── 🗄️ database/                # Database operations
│   └── database.py              # SQLAlchemy & CRUD operations
│
└── 💾 storage/                  # File storage
    ├── uploads/                 # Uploaded files
    ├── processed/               # Processed files
    ├── extracted_text/          # Extracted text
    ├── extracted_images/        # Extracted images
    ├── thumbnails/              # Document thumbnails
    └── document_packages/       # Packaged documents
```

## 🛠️ **Development**

### Adding New Features

1. **API Endpoints**: Add to `api/endpoints.py`
2. **Business Logic**: Create services in `services/`
3. **Data Models**: Define in `models/`
4. **Database Changes**: Update `supabase_schema.sql`

### Testing

```bash
# Test Supabase connection
python -c "from config.supabase_config import get_supabase_client; print('✅ Connection works')"

# Test embedding service
python -c "from services.embedding_service import EmbeddingService; EmbeddingService()"

```

> **Development guide**: See [DEVELOPMENT_GUIDE.md](./DEVELOPMENT_GUIDE.md)

## 🔧 **Troubleshooting**

### Common Issues

#### 🚫 ImportError: sentence_transformers
```bash
pip install --upgrade sentence-transformers transformers torch
```

#### 🚫 Supabase Connection Failed
1. Check `.env` file configuration
2. Verify Supabase project URL and keys
3. Ensure pgvector extension is enabled
4. Run the database schema

#### 🚫 OCR Not Working
1. Install Tesseract system dependency
2. Verify Tesseract is in PATH
3. For Windows: Add Tesseract to system PATH

#### 🚫 CUDA/GPU Issues
```bash
# Use CPU if no CUDA available
export HF_DEVICE=cpu
```

#### ⚠️ Langchain Deprecation Warnings
These warnings are normal and don't affect functionality. The code handles both old and new langchain imports automatically.

### Performance Optimization

- **GPU Usage**: Set `HF_DEVICE=cuda` for faster embedding generation
- **Chunk Size**: Adjust `EMBEDDING_CHUNK_SIZE` based on your document types
- **Similarity Threshold**: Lower values return more results but may be less relevant
- **Database Indexing**: Ensure proper vector indexes are created

## 📚 **Documentation**

- **[API Reference](./API_REFERENCE.md)**: Complete endpoint documentation
- **[Architecture Guide](./ARCHITECTURE.md)**: System design and components
- **[Development Guide](./DEVELOPMENT_GUIDE.md)**: Developer onboarding
- **[Deployment Guide](./DEPLOYMENT_GUIDE.md)**: Production deployment

## 🚀 **Production Deployment**

### Environment Variables
```env
# Production Supabase
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=your_production_anon_key
SUPABASE_SERVICE_KEY=your_production_service_key

# Performance
HF_DEVICE=cuda
EMBEDDING_CHUNK_SIZE=1000

# Security
# Add production configuration as needed
CORS_ORIGINS=["https://yourdomain.com"]
```

### Docker Deployment
```dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr tesseract-ocr-eng \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["python", "main.py"]
```

> **Complete deployment guide**: See [DEPLOYMENT_GUIDE.md](./DEPLOYMENT_GUIDE.md)

## 📊 **Usage Examples**

### Upload and Process Document
```bash
curl -X POST "http://localhost:8000/api/process-with-embeddings" \
  -F "file=@document.pdf" \
  -F "chunk_size=1000" \
  -F "chunk_overlap=200"
```

### Search Documents
```bash
curl -X POST "http://localhost:8000/api/search-documents" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "machine learning algorithms",
    "limit": 5,
    "similarity_threshold": 0.3
  }'
```

### Prepare Chat Context
```bash
curl -X POST "http://localhost:8000/api/chat/prepare-context" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is deep learning?",
    "use_context": true,
    "max_context_chunks": 8,
    "similarity_threshold": 0.3
  }'
```

## 🤝 **Contributing**

1. **Follow project structure** and conventions
2. **Add comprehensive tests** for new features
3. **Update documentation** for any changes
4. **Use proper error handling** and logging
5. **Follow security best practices**

## 📞 **Support**

For issues and questions:
1. Check the [Troubleshooting](#troubleshooting) section
2. Review the [API documentation](./API_REFERENCE.md)
3. Verify [configuration](#configuration) settings
4. Check [system dependencies](#system-dependencies)

## 📄 **License**

This project is licensed under the MIT License.

---

**Built with ❤️ using FastAPI, Supabase, HuggingFace Transformers, and CLIP**

> **Ready to get started?** Follow the [Quick Start](#quick-start) guide above!