# 🚀 Deployment Guide

Comprehensive production deployment guide for the Document Processing & AI Chat System.

## 📋 Table of Contents

- [Production Environment Setup](#production-environment-setup)
- [Docker Deployment](#docker-deployment)
- [Cloud Deployment Options](#cloud-deployment-options)
- [Environment Configuration](#environment-configuration)
- [Database Setup & Migration](#database-setup--migration)
- [Security Hardening](#security-hardening)
- [Monitoring & Logging](#monitoring--logging)
- [Performance Optimization](#performance-optimization)
- [Backup & Recovery](#backup--recovery)
- [CI/CD Pipeline](#cicd-pipeline)
- [Scaling Strategies](#scaling-strategies)
- [Troubleshooting Production Issues](#troubleshooting-production-issues)

---

## 🏭 Production Environment Setup

### Prerequisites

**Infrastructure Requirements:**
- **CPU**: 4+ cores recommended (8+ for high load)
- **RAM**: 8GB minimum (16GB+ recommended for GPU inference)
- **Storage**: 100GB+ SSD for application and logs
- **Network**: Stable internet for external API calls
- **GPU**: Optional but recommended for faster embeddings (NVIDIA GPU with CUDA support)

**Software Requirements:**
- **Python**: 3.11+
- **Docker**: 20.10+ (if using containerization)
- **Nginx**: For reverse proxy and load balancing
- **PostgreSQL**: 15+ with pgvector extension (or Supabase)
- **Redis**: For caching and session management (optional)

### System Dependencies

```bash
# Ubuntu/Debian production setup
sudo apt-get update && sudo apt-get upgrade -y

# Essential system packages
sudo apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3-pip \
    nginx \
    supervisor \
    tesseract-ocr \
    tesseract-ocr-eng \
    poppler-utils \
    build-essential \
    curl \
    git \
    htop \
    tmux

# For GPU support (if available)
sudo apt-get install -y nvidia-driver-525 nvidia-cuda-toolkit

# Create application user
sudo useradd -m -s /bin/bash appuser
sudo usermod -aG sudo appuser

# Create application directories
sudo mkdir -p /opt/document-processor
sudo chown appuser:appuser /opt/document-processor
```

### Application Setup

```bash
# Switch to application user
sudo su - appuser

# Clone and setup application
cd /opt/document-processor
git clone <repository-url> .

# Create production virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Install production-specific packages
pip install gunicorn uvicorn[standard] prometheus-client

# Create necessary directories
mkdir -p logs storage/{uploads,processed,extracted_text,extracted_images,thumbnails}
chmod 755 storage
```

---

## 🐳 Docker Deployment

### Production Dockerfile

```dockerfile
# Dockerfile
FROM python:3.11-slim-bullseye

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Create app user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Install system dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    poppler-utils \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create storage directories
RUN mkdir -p storage/{uploads,processed,extracted_text,extracted_images,thumbnails} logs

# Set permissions
RUN chown -R appuser:appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Run application
CMD ["gunicorn", "main:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]
```

### Docker Compose for Production

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  app:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: document-processor-app
    restart: unless-stopped
    environment:
      - ENVIRONMENT=production
      - SUPABASE_URL=${SUPABASE_URL}
      - SUPABASE_ANON_KEY=${SUPABASE_ANON_KEY}
      - SUPABASE_SERVICE_KEY=${SUPABASE_SERVICE_KEY}
      - HF_MODEL_NAME=${HF_MODEL_NAME}
      - HF_DEVICE=${HF_DEVICE}
    volumes:
      - ./storage:/app/storage
      - ./logs:/app/logs
    ports:
      - "127.0.0.1:8000:8000"
    networks:
      - app-network
    depends_on:
      - redis
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  redis:
    image: redis:7-alpine
    container_name: document-processor-redis
    restart: unless-stopped
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
    networks:
      - app-network
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 30s
      timeout: 10s
      retries: 3

  nginx:
    image: nginx:alpine
    container_name: document-processor-nginx
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
      - ./logs/nginx:/var/log/nginx
    networks:
      - app-network
    depends_on:
      - app

volumes:
  redis_data:

networks:
  app-network:
    driver: bridge
```

### Nginx Configuration

```nginx
# nginx.conf
events {
    worker_connections 1024;
}

http {
    include       /etc/nginx/mime.types;
    default_type  application/octet-stream;

    # Logging
    log_format main '$remote_addr - $remote_user [$time_local] "$request" '
                    '$status $body_bytes_sent "$http_referer" '
                    '"$http_user_agent" "$http_x_forwarded_for" '
                    'rt=$request_time uct="$upstream_connect_time" '
                    'uht="$upstream_header_time" urt="$upstream_response_time"';

    access_log /var/log/nginx/access.log main;
    error_log /var/log/nginx/error.log warn;

    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;
    add_header Content-Security-Policy "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline';" always;

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req_zone $binary_remote_addr zone=upload:10m rate=2r/m;

    # Upstream backend
    upstream app_backend {
        server app:8000;
        keepalive 32;
    }

    # Main server block
    server {
        listen 80;
        server_name your-domain.com www.your-domain.com;
        
        # Redirect HTTP to HTTPS
        return 301 https://$host$request_uri;
    }

    server {
        listen 443 ssl http2;
        server_name your-domain.com www.your-domain.com;

        # SSL Configuration
        ssl_certificate /etc/nginx/ssl/cert.pem;
        ssl_certificate_key /etc/nginx/ssl/key.pem;
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384;
        ssl_prefer_server_ciphers off;
        ssl_session_cache shared:SSL:10m;
        ssl_session_timeout 10m;

        # File upload limits
        client_max_body_size 100M;
        client_body_timeout 60s;
        client_header_timeout 60s;

        # Compression
        gzip on;
        gzip_vary on;
        gzip_min_length 1024;
        gzip_types text/plain text/css text/xml text/javascript application/javascript application/xml+rss application/json;

        # Health check endpoint (no rate limiting)
        location /health {
            proxy_pass http://app_backend;
            access_log off;
        }


        # File upload endpoints (limited rate)
        location ~ ^/api/(upload|process) {
            limit_req zone=upload burst=5 nodelay;
            
            proxy_pass http://app_backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            proxy_connect_timeout 60s;
            proxy_send_timeout 300s;
            proxy_read_timeout 300s;
            proxy_request_buffering off;
        }

        # API endpoints (standard rate limiting)
        location /api/ {
            limit_req zone=api burst=20 nodelay;
            
            proxy_pass http://app_backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            proxy_connect_timeout 30s;
            proxy_send_timeout 60s;
            proxy_read_timeout 60s;
        }

        # Documentation
        location /docs {
            proxy_pass http://app_backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }

        # Default location
        location / {
            proxy_pass http://app_backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
    }
}
```

---

## ☁️ Cloud Deployment Options

### 1. AWS Deployment

#### **EC2 + RDS + S3 Setup**

```bash
# EC2 instance setup (Ubuntu 22.04 LTS)
# Instance type: t3.large or higher (c5.xlarge for GPU workloads)

# User data script for EC2
#!/bin/bash
yum update -y
yum install -y docker
systemctl start docker
systemctl enable docker
usermod -a -G docker ec2-user

# Install docker-compose
curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose

# Setup application
mkdir -p /opt/document-processor
cd /opt/document-processor
# Deploy your application here
```

#### **ECS Fargate Deployment**

```yaml
# ecs-task-definition.json
{
  "family": "document-processor",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "2048",
  "memory": "4096",
  "executionRoleArn": "arn:aws:iam::ACCOUNT:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::ACCOUNT:role/ecsTaskRole",
  "containerDefinitions": [
    {
      "name": "document-processor",
      "image": "your-account.dkr.ecr.region.amazonaws.com/document-processor:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {"name": "ENVIRONMENT", "value": "production"},
        {"name": "HF_DEVICE", "value": "cpu"}
      ],
      "secrets": [
        {"name": "SUPABASE_URL", "valueFrom": "arn:aws:secretsmanager:region:account:secret:prod/supabase-url"}
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/document-processor",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      },
      "healthCheck": {
        "command": ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"],
        "interval": 30,
        "timeout": 5,
        "retries": 3,
        "startPeriod": 60
      }
    }
  ]
}
```

### 2. Google Cloud Platform

#### **Cloud Run Deployment**

```yaml
# cloudrun.yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: document-processor
  annotations:
    run.googleapis.com/ingress: all
    run.googleapis.com/execution-environment: gen2
spec:
  template:
    metadata:
      annotations:
        run.googleapis.com/cpu-throttling: "false"
        run.googleapis.com/memory: "4Gi"
        run.googleapis.com/cpu: "2000m"
        run.googleapis.com/max-scale: "10"
        run.googleapis.com/min-scale: "1"
    spec:
      containerConcurrency: 100
      timeoutSeconds: 300
      containers:
      - image: gcr.io/PROJECT_ID/document-processor:latest
        ports:
        - containerPort: 8000
        env:
        - name: ENVIRONMENT
          value: "production"
        - name: SUPABASE_URL
          valueFrom:
            secretKeyRef:
              name: supabase-url
              key: url
        resources:
          limits:
            cpu: "2000m"
            memory: "4Gi"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
```

### 3. Digital Ocean App Platform

```yaml
# .do/app.yaml
name: document-processor
services:
- name: api
  source_dir: /
  dockerfile_path: Dockerfile
  github:
    repo: your-username/document-processor
    branch: main
  http_port: 8000
  instance_count: 2
  instance_size_slug: professional-s
  routes:
  - path: /
  health_check:
    http_path: /health
    initial_delay_seconds: 30
    period_seconds: 10
    timeout_seconds: 5
    success_threshold: 1
    failure_threshold: 3
  envs:
  - key: ENVIRONMENT
    value: production
  - key: SUPABASE_URL
    value: ${SUPABASE_URL}
    type: SECRET
  - key: SUPABASE_ANON_KEY
    value: ${SUPABASE_ANON_KEY}
    type: SECRET
  - key: HF_DEVICE
    value: cpu
```

---

## ⚙️ Environment Configuration

### Production Environment Variables

```bash
# .env.production
# Application Settings
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO

# Server Configuration
HOST=0.0.0.0
PORT=8000
WORKERS=4

# Database Configuration
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=your_production_anon_key
SUPABASE_SERVICE_KEY=your_production_service_key


# AI Model Configuration
HF_MODEL_NAME=sentence-transformers/all-mpnet-base-v2
HF_DEVICE=cuda  # or 'cpu' if no GPU
HF_CACHE_DIR=/opt/models
EMBEDDING_CHUNK_SIZE=1000
EMBEDDING_CHUNK_OVERLAP=200
EMBEDDING_DIMENSIONS=768

# File Storage Configuration
UPLOAD_DIR=/opt/document-processor/storage/uploads
MAX_FILE_SIZE_MB=50
ALLOWED_FILE_TYPES=pdf,docx,xlsx,pptx,png,jpg,jpeg,txt

# Caching Configuration
REDIS_URL=redis://localhost:6379/0
CACHE_TTL_SECONDS=3600

# Rate Limiting
RATE_LIMIT_PER_MINUTE=60
RATE_LIMIT_BURST=100

# Monitoring
ENABLE_METRICS=true
METRICS_PORT=9090
SENTRY_DSN=your_sentry_dsn_for_error_tracking

# CORS Configuration
CORS_ORIGINS=["https://yourdomain.com","https://app.yourdomain.com"]
CORS_ALLOW_CREDENTIALS=true

# SSL Configuration
SSL_CERT_PATH=/etc/ssl/certs/cert.pem
SSL_KEY_PATH=/etc/ssl/private/key.pem

# Backup Configuration
BACKUP_ENABLED=true
BACKUP_SCHEDULE="0 2 * * *"  # Daily at 2 AM
BACKUP_RETENTION_DAYS=30
```

### Configuration Validation

```python
# config/settings.py
from pydantic import BaseSettings, validator
from typing import List, Optional
import os

class Settings(BaseSettings):
    """Production settings with validation"""
    
    # Application
    environment: str = "production"
    debug: bool = False
    log_level: str = "INFO"
    
    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    
    # Database
    supabase_url: str
    supabase_anon_key: str
    supabase_service_key: str
    
    
    # AI Models
    hf_model_name: str = "sentence-transformers/all-mpnet-base-v2"
    hf_device: str = "cpu"
    
    # File Storage
    upload_dir: str = "./storage/uploads"
    max_file_size_mb: int = 50
    
    # CORS
    cors_origins: List[str] = ["*"]
    
    
    @validator('supabase_url')
    def validate_supabase_url(cls, v):
        if not v.startswith('https://'):
            raise ValueError('Supabase URL must use HTTPS in production')
        return v
    
    @validator('cors_origins')
    def validate_cors_origins(cls, v):
        if "*" in v and len(v) > 1:
            raise ValueError('CORS origins cannot contain "*" with other origins')
        return v
    
    class Config:
        env_file = ".env"
        case_sensitive = False

# Usage
settings = Settings()
```

---

## 🗄️ Database Setup & Migration

### Supabase Production Setup

```sql
-- Production database optimization
-- supabase_production_schema.sql

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Optimized tables with production considerations


-- Enhanced document metadata table
CREATE TABLE documents_metadata (
    document_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    original_filename VARCHAR(255) NOT NULL,
    file_type VARCHAR(50) NOT NULL,
    file_size BIGINT NOT NULL,
    file_hash VARCHAR(64) UNIQUE, -- SHA-256 for deduplication
    storage_path TEXT NOT NULL,
    total_chunks INTEGER DEFAULT 0,
    total_characters INTEGER DEFAULT 0,
    processing_status VARCHAR(50) DEFAULT 'pending',
    embedding_status VARCHAR(50) DEFAULT 'pending',
    error_message TEXT,
    processing_started_at TIMESTAMP WITH TIME ZONE,
    processing_completed_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Production indexes for documents
CREATE INDEX CONCURRENTLY idx_documents_status ON documents_metadata(processing_status);
CREATE INDEX CONCURRENTLY idx_documents_file_hash ON documents_metadata(file_hash);
CREATE INDEX CONCURRENTLY idx_documents_created_at ON documents_metadata(created_at);

-- Optimized embeddings table with partitioning
CREATE TABLE document_embeddings (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID REFERENCES documents_metadata(document_id) ON DELETE CASCADE,
    chunk_id UUID DEFAULT uuid_generate_v4(),
    chunk_text TEXT NOT NULL,
    embedding vector(768) NOT NULL,
    chunk_index INTEGER NOT NULL,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
) PARTITION BY HASH (document_id);

-- Create partitions for better performance
CREATE TABLE document_embeddings_0 PARTITION OF document_embeddings
    FOR VALUES WITH (modulus 4, remainder 0);
CREATE TABLE document_embeddings_1 PARTITION OF document_embeddings
    FOR VALUES WITH (modulus 4, remainder 1);
CREATE TABLE document_embeddings_2 PARTITION OF document_embeddings
    FOR VALUES WITH (modulus 4, remainder 2);
CREATE TABLE document_embeddings_3 PARTITION OF document_embeddings
    FOR VALUES WITH (modulus 4, remainder 3);

-- Vector similarity indexes for each partition
CREATE INDEX CONCURRENTLY idx_embeddings_vector_0 ON document_embeddings_0 
    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX CONCURRENTLY idx_embeddings_vector_1 ON document_embeddings_1 
    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX CONCURRENTLY idx_embeddings_vector_2 ON document_embeddings_2 
    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX CONCURRENTLY idx_embeddings_vector_3 ON document_embeddings_3 
    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);



-- Database functions for production
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Triggers for timestamp updates
CREATE TRIGGER update_documents_updated_at BEFORE UPDATE ON documents_metadata
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
```

### Database Migration Script

```python
# scripts/migrate_database.py
import asyncio
import os
import logging
from supabase import create_client
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseMigrator:
    def __init__(self):
        self.supabase = create_client(
            os.getenv("SUPABASE_URL"),
            os.getenv("SUPABASE_SERVICE_KEY")
        )
    
    async def run_migration(self, migration_file: str):
        """Run a single migration file"""
        try:
            with open(migration_file, 'r') as f:
                migration_sql = f.read()
            
            # Split on semicolons and execute each statement
            statements = [s.strip() for s in migration_sql.split(';') if s.strip()]
            
            for statement in statements:
                if statement:
                    logger.info(f"Executing: {statement[:100]}...")
                    response = self.supabase.rpc('exec_sql', {'sql': statement}).execute()
                    logger.info("Statement executed successfully")
            
            logger.info(f"Migration {migration_file} completed successfully")
            
        except Exception as e:
            logger.error(f"Migration failed: {e}")
            raise
    
    async def run_all_migrations(self):
        """Run all pending migrations"""
        migrations_dir = Path("migrations")
        migration_files = sorted(migrations_dir.glob("*.sql"))
        
        for migration_file in migration_files:
            logger.info(f"Running migration: {migration_file}")
            await self.run_migration(migration_file)
        
        logger.info("All migrations completed successfully")

if __name__ == "__main__":
    migrator = DatabaseMigrator()
    asyncio.run(migrator.run_all_migrations())
```

---

## 🔒 Security Hardening

### 1. Application Security

```python
# security/security_middleware.py
from fastapi import Request, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import time
import hashlib
from collections import defaultdict, deque
from typing import Dict, Deque
import logging

logger = logging.getLogger(__name__)

class SecurityMiddleware:
    """Production security middleware"""
    
    def __init__(self):
        # Rate limiting storage
        self.rate_limiters: Dict[str, Deque] = defaultdict(deque)
        self.blocked_ips: Dict[str, float] = {}
        
        # Security headers
        self.security_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Content-Security-Policy": "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline';",
            "Permissions-Policy": "camera=(), microphone=(), geolocation=()"
        }
    
    async def __call__(self, request: Request, call_next):
        """Process request with security checks"""
        client_ip = self.get_client_ip(request)
        
        # Check if IP is blocked
        if self.is_ip_blocked(client_ip):
            raise HTTPException(status_code=429, detail="IP temporarily blocked")
        
        # Rate limiting
        if not self.check_rate_limit(client_ip, request.url.path):
            self.block_ip_temporarily(client_ip)
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        
        # Process request
        response = await call_next(request)
        
        # Add security headers
        for header, value in self.security_headers.items():
            response.headers[header] = value
        
        return response
    
    def get_client_ip(self, request: Request) -> str:
        """Get real client IP considering proxy headers"""
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        return request.client.host
    
    def check_rate_limit(self, client_ip: str, path: str, limit: int = 60) -> bool:
        """Check rate limit for client IP"""
        now = time.time()
        window = 60  # 1 minute window
        
        # Clean old entries
        rate_limiter = self.rate_limiters[client_ip]
        while rate_limiter and rate_limiter[0] < now - window:
            rate_limiter.popleft()
        
        # Check limit
        if len(rate_limiter) >= limit:
            logger.warning(f"Rate limit exceeded for IP {client_ip} on path {path}")
            return False
        
        # Add current request
        rate_limiter.append(now)
        return True
    
    def is_ip_blocked(self, client_ip: str) -> bool:
        """Check if IP is temporarily blocked"""
        if client_ip in self.blocked_ips:
            if time.time() < self.blocked_ips[client_ip]:
                return True
            else:
                del self.blocked_ips[client_ip]
        return False
    
    def block_ip_temporarily(self, client_ip: str, duration: int = 900):
        """Block IP temporarily (15 minutes default)"""
        self.blocked_ips[client_ip] = time.time() + duration
        logger.warning(f"IP {client_ip} blocked temporarily for {duration} seconds")

```

### 2. File Upload Security

```python
# security/file_security.py
import magic
import hashlib
import os
from pathlib import Path
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

class FileSecurityValidator:
    """Secure file validation and processing"""
    
    ALLOWED_MIME_TYPES = {
        'application/pdf',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        'application/vnd.openxmlformats-officedocument.presentationml.presentation',
        'image/png',
        'image/jpeg',
        'text/plain'
    }
    
    DANGEROUS_EXTENSIONS = {
        '.exe', '.bat', '.cmd', '.com', '.pif', '.scr', '.vbs', '.js',
        '.jar', '.sh', '.ps1', '.php', '.asp', '.aspx', '.jsp'
    }
    
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
    
    @classmethod
    def validate_file(cls, file_path: str, original_filename: str) -> dict:
        """Comprehensive file validation"""
        try:
            path = Path(file_path)
            
            # Check file size
            file_size = path.stat().st_size
            if file_size > cls.MAX_FILE_SIZE:
                raise ValueError(f"File size {file_size} exceeds maximum {cls.MAX_FILE_SIZE}")
            
            # Check file extension
            file_ext = path.suffix.lower()
            if file_ext in cls.DANGEROUS_EXTENSIONS:
                raise ValueError(f"Dangerous file extension: {file_ext}")
            
            # Check MIME type using python-magic
            mime_type = magic.from_file(file_path, mime=True)
            if mime_type not in cls.ALLOWED_MIME_TYPES:
                raise ValueError(f"Unsupported MIME type: {mime_type}")
            
            # Calculate file hash for deduplication
            file_hash = cls.calculate_file_hash(file_path)
            
            # Additional security checks
            cls.check_file_content_security(file_path, mime_type)
            
            return {
                "valid": True,
                "mime_type": mime_type,
                "file_size": file_size,
                "file_hash": file_hash,
                "sanitized_filename": cls.sanitize_filename(original_filename)
            }
            
        except Exception as e:
            logger.error(f"File validation failed for {file_path}: {e}")
            return {
                "valid": False,
                "error": str(e)
            }
    
    @staticmethod
    def calculate_file_hash(file_path: str) -> str:
        """Calculate SHA-256 hash of file"""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitize filename for safe storage"""
        # Remove path traversal attempts
        filename = os.path.basename(filename)
        
        # Remove or replace dangerous characters
        dangerous_chars = '<>:"/\\|?*'
        for char in dangerous_chars:
            filename = filename.replace(char, '_')
        
        # Limit length
        if len(filename) > 255:
            name, ext = os.path.splitext(filename)
            filename = name[:255-len(ext)] + ext
        
        return filename
    
    @staticmethod
    def check_file_content_security(file_path: str, mime_type: str):
        """Additional content-based security checks"""
        
        if mime_type == 'application/pdf':
            # Check for malicious PDF content
            with open(file_path, 'rb') as f:
                content = f.read(1024)  # Check first 1KB
                if b'/JavaScript' in content or b'/JS' in content:
                    raise ValueError("PDF contains potentially dangerous JavaScript")
        
        elif mime_type.startswith('image/'):
            # Check image file headers
            with open(file_path, 'rb') as f:
                header = f.read(10)
                if mime_type == 'image/jpeg' and not header.startswith(b'\xFF\xD8\xFF'):
                    raise ValueError("Invalid JPEG file header")
                elif mime_type == 'image/png' and not header.startswith(b'\x89PNG\r\n\x1a\n'):
                    raise ValueError("Invalid PNG file header")
```

### 3. Environment Security

```bash
#!/bin/bash
# scripts/security_hardening.sh

# Production security hardening script

echo "🔒 Starting security hardening..."

# 1. Update system
sudo apt-get update && sudo apt-get upgrade -y

# 2. Configure firewall
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow ssh
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw --force enable

# 3. SSH hardening
sudo sed -i 's/#PermitRootLogin yes/PermitRootLogin no/' /etc/ssh/sshd_config
sudo sed -i 's/#PasswordAuthentication yes/PasswordAuthentication no/' /etc/ssh/sshd_config
sudo sed -i 's/#PubkeyAuthentication yes/PubkeyAuthentication yes/' /etc/ssh/sshd_config
sudo systemctl restart sshd

# 4. Install fail2ban
sudo apt-get install -y fail2ban
sudo systemctl enable fail2ban
sudo systemctl start fail2ban

# 5. Configure fail2ban for nginx
sudo tee /etc/fail2ban/jail.local << EOF
[DEFAULT]
bantime = 3600
findtime = 600
maxretry = 5

[nginx-http-auth]
enabled = true
port = http,https
logpath = /var/log/nginx/error.log

[nginx-noscript]
enabled = true
port = http,https
logpath = /var/log/nginx/access.log
maxretry = 6

[nginx-badbots]
enabled = true
port = http,https
logpath = /var/log/nginx/access.log
maxretry = 2

[nginx-noproxy]
enabled = true
port = http,https
logpath = /var/log/nginx/access.log
maxretry = 2
EOF

sudo systemctl restart fail2ban

# 6. Set up automatic security updates
sudo apt-get install -y unattended-upgrades
sudo dpkg-reconfigure -plow unattended-upgrades

# 7. Configure file permissions
sudo chmod 750 /opt/document-processor
sudo chmod 640 /opt/document-processor/.env
sudo chown -R appuser:appuser /opt/document-processor

# 8. Set up log rotation
sudo tee /etc/logrotate.d/document-processor << EOF
/opt/document-processor/logs/*.log {
    daily
    missingok
    rotate 52
    compress
    delaycompress
    notifempty
    create 644 appuser appuser
}
EOF

echo "✅ Security hardening completed!"
```

---

## 📊 Monitoring & Logging

### 1. Application Monitoring

```python
# monitoring/metrics.py
from prometheus_client import Counter, Histogram, Gauge, Info, start_http_server
import time
import logging
from functools import wraps
import psutil
import asyncio

# Metrics definition
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration')
ACTIVE_USERS = Gauge('active_users_total', 'Number of active users')
EMBEDDING_GENERATION_TIME = Histogram('embedding_generation_seconds', 'Time to generate embeddings')
DOCUMENT_PROCESSING_TIME = Histogram('document_processing_seconds', 'Time to process documents')
SEARCH_QUERY_TIME = Histogram('search_query_seconds', 'Time to execute search queries')
CHAT_CONTEXT_TIME = Histogram('chat_context_preparation_seconds', 'Time to prepare chat context')

# System metrics
CPU_USAGE = Gauge('system_cpu_usage_percent', 'CPU usage percentage')
MEMORY_USAGE = Gauge('system_memory_usage_bytes', 'Memory usage in bytes')
DISK_USAGE = Gauge('system_disk_usage_bytes', 'Disk usage in bytes')

# Database metrics
DB_CONNECTIONS = Gauge('database_connections_active', 'Active database connections')
DB_QUERY_TIME = Histogram('database_query_seconds', 'Database query execution time')

# Application info
APP_INFO = Info('application_info', 'Application information')
APP_INFO.info({
    'version': '1.0.0',
    'python_version': '3.11',
    'environment': 'production'
})

class MetricsCollector:
    """Centralized metrics collection"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._monitoring_active = False
    
    def start_monitoring(self, port: int = 9090):
        """Start Prometheus metrics server"""
        try:
            start_http_server(port)
            self._monitoring_active = True
            self.logger.info(f"Metrics server started on port {port}")
            
            # Start background system monitoring
            asyncio.create_task(self._monitor_system_metrics())
            
        except Exception as e:
            self.logger.error(f"Failed to start metrics server: {e}")
    
    async def _monitor_system_metrics(self):
        """Monitor system metrics in background"""
        while self._monitoring_active:
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                CPU_USAGE.set(cpu_percent)
                
                # Memory usage
                memory = psutil.virtual_memory()
                MEMORY_USAGE.set(memory.used)
                
                # Disk usage
                disk = psutil.disk_usage('/')
                DISK_USAGE.set(disk.used)
                
                await asyncio.sleep(30)  # Update every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Error collecting system metrics: {e}")
                await asyncio.sleep(60)  # Wait longer on error

def monitor_performance(metric: Histogram = None):
    """Decorator to monitor function performance"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                if metric:
                    metric.observe(duration)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                if metric:
                    metric.observe(duration)
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator

# Usage examples
@monitor_performance(EMBEDDING_GENERATION_TIME)
async def generate_embedding(text: str):
    """Generate embedding with performance monitoring"""
    pass

@monitor_performance(DOCUMENT_PROCESSING_TIME)
async def process_document(file_path: str):
    """Process document with performance monitoring"""
    pass
```

### 2. Centralized Logging

```python
# monitoring/logging_setup.py
import logging
import logging.config
import json
import sys
from datetime import datetime
from pathlib import Path
import os

class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def format(self, record):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in log_entry and not key.startswith('_'):
                log_entry[key] = value
        
        return json.dumps(log_entry)

class ProductionLoggingConfig:
    """Production logging configuration"""
    
    @staticmethod
    def setup_logging():
        log_dir = Path("/var/log/document-processor")
        log_dir.mkdir(exist_ok=True)
        
        config = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "json": {
                    "()": JSONFormatter
                },
                "detailed": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "level": "INFO",
                    "formatter": "detailed",
                    "stream": sys.stdout
                },
                "file_json": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "level": "DEBUG",
                    "formatter": "json",
                    "filename": f"{log_dir}/application.json",
                    "maxBytes": 50 * 1024 * 1024,  # 50MB
                    "backupCount": 10
                },
                "error_file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "level": "ERROR",
                    "formatter": "json",
                    "filename": f"{log_dir}/errors.json",
                    "maxBytes": 50 * 1024 * 1024,
                    "backupCount": 20
                },
                "audit_file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "level": "INFO",
                    "formatter": "json",
                    "filename": f"{log_dir}/audit.json",
                    "maxBytes": 100 * 1024 * 1024,  # 100MB
                    "backupCount": 50
                }
            },
            "loggers": {
                "": {  # Root logger
                    "level": "INFO",
                    "handlers": ["console", "file_json", "error_file"]
                },
                "api": {
                    "level": "INFO",
                    "handlers": ["file_json", "audit_file"],
                    "propagate": False
                },
                "services": {
                    "level": "DEBUG",
                    "handlers": ["file_json"],
                    "propagate": False
                },
                "security": {
                    "level": "WARNING",
                    "handlers": ["error_file", "audit_file"],
                    "propagate": False
                }
            }
        }
        
        logging.config.dictConfig(config)

# Usage in application
class AuditLogger:
    """Audit logging for security events"""
    
    def __init__(self):
        self.logger = logging.getLogger("security.audit")
    
    
    def log_document_access(self, document_id: str, action: str):
        """Log document access"""
        self.logger.info("Document access", extra={
            "event_type": "document_access",
            "document_id": document_id,
            "action": action,
            "timestamp": datetime.utcnow().isoformat()
        })
```

### 3. Health Checks

```python
# monitoring/health_checks.py
import asyncio
import time
from typing import Dict, Any
import logging
from dataclasses import dataclass
from enum import Enum

class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"

@dataclass
class HealthCheckResult:
    status: HealthStatus
    message: str
    details: Dict[str, Any] = None
    duration_ms: float = 0

class ComprehensiveHealthChecker:
    """Comprehensive health checking system"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.checks = {}
    
    def register_check(self, name: str, check_func, timeout: int = 30):
        """Register a health check"""
        self.checks[name] = {
            "func": check_func,
            "timeout": timeout
        }
    
    async def run_all_checks(self) -> Dict[str, HealthCheckResult]:
        """Run all registered health checks"""
        results = {}
        
        for name, check_config in self.checks.items():
            try:
                start_time = time.time()
                
                # Run check with timeout
                result = await asyncio.wait_for(
                    check_config["func"](),
                    timeout=check_config["timeout"]
                )
                
                duration = (time.time() - start_time) * 1000
                result.duration_ms = duration
                results[name] = result
                
            except asyncio.TimeoutError:
                results[name] = HealthCheckResult(
                    status=HealthStatus.UNHEALTHY,
                    message=f"Health check timed out after {check_config['timeout']}s"
                )
            except Exception as e:
                self.logger.error(f"Health check {name} failed: {e}")
                results[name] = HealthCheckResult(
                    status=HealthStatus.UNHEALTHY,
                    message=str(e)
                )
        
        return results
    
    async def get_overall_status(self) -> HealthCheckResult:
        """Get overall system health status"""
        results = await self.run_all_checks()
        
        # Determine overall status
        unhealthy_count = sum(1 for r in results.values() if r.status == HealthStatus.UNHEALTHY)
        degraded_count = sum(1 for r in results.values() if r.status == HealthStatus.DEGRADED)
        
        if unhealthy_count > 0:
            overall_status = HealthStatus.UNHEALTHY
            message = f"{unhealthy_count} critical systems are unhealthy"
        elif degraded_count > 0:
            overall_status = HealthStatus.DEGRADED
            message = f"{degraded_count} systems are degraded"
        else:
            overall_status = HealthStatus.HEALTHY
            message = "All systems are healthy"
        
        return HealthCheckResult(
            status=overall_status,
            message=message,
            details={
                "checks": {name: result.__dict__ for name, result in results.items()},
                "summary": {
                    "total_checks": len(results),
                    "healthy": sum(1 for r in results.values() if r.status == HealthStatus.HEALTHY),
                    "degraded": degraded_count,
                    "unhealthy": unhealthy_count
                }
            }
        )

# Specific health checks
async def check_database_connection() -> HealthCheckResult:
    """Check database connectivity"""
    try:
        from config.supabase_config import get_supabase_client
        
        client = get_supabase_client()
        response = client.table("users").select("id").limit(1).execute()
        
        return HealthCheckResult(
            status=HealthStatus.HEALTHY,
            message="Database connection successful"
        )
    except Exception as e:
        return HealthCheckResult(
            status=HealthStatus.UNHEALTHY,
            message=f"Database connection failed: {e}"
        )

async def check_embedding_service() -> HealthCheckResult:
    """Check embedding service"""
    try:
        from services.embedding_service import EmbeddingService
        
        service = EmbeddingService()
        await service.initialize()
        
        # Test embedding generation
        test_embedding = await service.generate_embedding("test")
        
        if len(test_embedding) == 768:
            return HealthCheckResult(
                status=HealthStatus.HEALTHY,
                message="Embedding service working correctly"
            )
        else:
            return HealthCheckResult(
                status=HealthStatus.DEGRADED,
                message=f"Unexpected embedding dimension: {len(test_embedding)}"
            )
            
    except Exception as e:
        return HealthCheckResult(
            status=HealthStatus.UNHEALTHY,
            message=f"Embedding service failed: {e}"
        )

async def check_disk_space() -> HealthCheckResult:
    """Check available disk space"""
    try:
        import shutil
        
        total, used, free = shutil.disk_usage("/")
        free_percentage = (free / total) * 100
        
        if free_percentage > 20:
            status = HealthStatus.HEALTHY
            message = f"Disk space healthy: {free_percentage:.1f}% free"
        elif free_percentage > 10:
            status = HealthStatus.DEGRADED
            message = f"Disk space low: {free_percentage:.1f}% free"
        else:
            status = HealthStatus.UNHEALTHY
            message = f"Disk space critical: {free_percentage:.1f}% free"
        
        return HealthCheckResult(
            status=status,
            message=message,
            details={
                "total_gb": total / (1024**3),
                "used_gb": used / (1024**3),
                "free_gb": free / (1024**3),
                "free_percentage": free_percentage
            }
        )
        
    except Exception as e:
        return HealthCheckResult(
            status=HealthStatus.UNHEALTHY,
            message=f"Disk space check failed: {e}"
        )

# Setup health checker
health_checker = ComprehensiveHealthChecker()
health_checker.register_check("database", check_database_connection, timeout=10)
health_checker.register_check("embedding_service", check_embedding_service, timeout=30)
health_checker.register_check("disk_space", check_disk_space, timeout=5)
```

---

## ⚡ Performance Optimization

### 1. Application Performance

```python
# optimization/performance_optimizer.py
import asyncio
import time
from functools import wraps, lru_cache
from typing import Any, Dict, List, Optional
import pickle
import redis
import logging

logger = logging.getLogger(__name__)

class PerformanceOptimizer:
    """Performance optimization utilities"""
    
    def __init__(self):
        self.redis_client = redis.Redis.from_url(
            os.getenv("REDIS_URL", "redis://localhost:6379/0")
        )
    
    def cached_result(self, ttl: int = 3600, key_prefix: str = "cache"):
        """Cache function results in Redis"""
        def decorator(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                # Generate cache key
                cache_key = f"{key_prefix}:{func.__name__}:{hash(str(args) + str(kwargs))}"
                
                try:
                    # Try to get from cache
                    cached = self.redis_client.get(cache_key)
                    if cached:
                        logger.debug(f"Cache hit for {cache_key}")
                        return pickle.loads(cached)
                    
                    # Not in cache, compute result
                    result = await func(*args, **kwargs)
                    
                    # Store in cache
                    self.redis_client.setex(
                        cache_key, 
                        ttl, 
                        pickle.dumps(result)
                    )
                    logger.debug(f"Cached result for {cache_key}")
                    
                    return result
                    
                except Exception as e:
                    logger.warning(f"Cache operation failed for {cache_key}: {e}")
                    # Fallback to direct computation
                    return await func(*args, **kwargs)
            
            return async_wrapper
        return decorator
    
    @staticmethod
    def batch_processor(batch_size: int = 32):
        """Process items in batches for better performance"""
        def decorator(func):
            @wraps(func)
            async def wrapper(items: List[Any], *args, **kwargs):
                results = []
                
                for i in range(0, len(items), batch_size):
                    batch = items[i:i + batch_size]
                    batch_results = await func(batch, *args, **kwargs)
                    results.extend(batch_results)
                    
                    # Allow other tasks to run
                    await asyncio.sleep(0)
                
                return results
            
            return wrapper
        return decorator

# Example usage
performance_optimizer = PerformanceOptimizer()

class OptimizedEmbeddingService:
    """Optimized embedding service with caching and batching"""
    
    def __init__(self):
        self.model = None
        self.performance_optimizer = PerformanceOptimizer()
    
    @performance_optimizer.cached_result(ttl=7200, key_prefix="embedding")
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding with caching"""
        if not self.model:
            await self.initialize()
        
        embedding = self.model.encode(text, convert_to_tensor=False)
        return embedding.tolist()
    
    @performance_optimizer.batch_processor(batch_size=32)
    async def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings in batches"""
        if not self.model:
            await self.initialize()
        
        embeddings = self.model.encode(
            texts,
            batch_size=len(texts),
            show_progress_bar=False,
            convert_to_tensor=False
        )
        
        return embeddings.tolist()

# Database query optimization
class OptimizedDatabaseQueries:
    """Optimized database queries with connection pooling"""
    
    def __init__(self):
        self.connection_pool = None
    
    async def optimized_similarity_search(
        self,
        query_embedding: List[float],
        limit: int = 8,
        threshold: float = 0.3
    ) -> List[Dict[str, Any]]:
        """Optimized similarity search with proper indexing"""
        
        # Use prepared statement for better performance
        query = """
        SELECT 
            de.document_id,
            de.chunk_text,
            de.chunk_index,
            dm.original_filename,
            1 - (de.embedding <=> $1::vector) as similarity_score
        FROM document_embeddings de
        JOIN documents_metadata dm ON de.document_id = dm.document_id
        WHERE de.embedding <=> $1::vector < $2
        ORDER BY de.embedding <=> $1::vector
        LIMIT $3;
        """
        
        # Execute with proper parameter binding
        # This would use your actual database client
        # results = await self.execute_query(query, query_embedding, 1-threshold, limit)
        
        return []  # Placeholder

# Memory optimization
class MemoryOptimizer:
    """Memory usage optimization"""
    
    @staticmethod
    def memory_efficient_file_processing(file_path: str, chunk_size: int = 8192):
        """Process large files in chunks to manage memory"""
        def file_chunk_generator():
            with open(file_path, 'rb') as f:
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    yield chunk
        
        return file_chunk_generator()
    
    @staticmethod
    async def cleanup_memory():
        """Force garbage collection and memory cleanup"""
        import gc
        
        # Force garbage collection
        collected = gc.collect()
        
        # Clear any caches that might be holding memory
        # (Application-specific cleanup)
        
        logger.info(f"Memory cleanup completed, collected {collected} objects")
        
        return collected
```

### 2. Database Optimization

```sql
-- Database performance optimization queries
-- performance_optimization.sql

-- Optimize vector search performance
SET maintenance_work_mem = '1GB';
SET max_parallel_maintenance_workers = 4;

-- Create optimized indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_embeddings_vector_optimized 
ON document_embeddings 
USING ivfflat (embedding vector_cosine_ops) 
WITH (lists = 1000);


-- Composite indexes for common query patterns
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_documents_status 
ON documents_metadata(processing_status) 
WHERE processing_status = 'completed';

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_documents_created 
ON documents_metadata(created_at DESC);

-- Optimize search analytics
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_searches_timestamp 
ON similarity_searches(search_timestamp DESC);

-- Partitioning for large tables (if needed)
-- CREATE TABLE document_embeddings_archive_2024 PARTITION OF document_embeddings
-- FOR VALUES FROM ('2024-01-01') TO ('2025-01-01');

-- Query optimization function
CREATE OR REPLACE FUNCTION optimized_similarity_search(
    query_embedding vector(768),
    similarity_threshold float DEFAULT 0.3,
    result_limit int DEFAULT 8
)
RETURNS TABLE (
    document_id uuid,
    chunk_text text,
    similarity_score float,
    original_filename text,
    chunk_index int
)
LANGUAGE plpgsql
AS $$
BEGIN
    -- Set local configuration for this query
    SET LOCAL work_mem = '256MB';
    SET LOCAL effective_cache_size = '4GB';
    
    RETURN QUERY
    SELECT 
        de.document_id,
        de.chunk_text,
        1 - (de.embedding <=> query_embedding) as similarity_score,
        dm.original_filename,
        de.chunk_index
    FROM document_embeddings de
    JOIN documents_metadata dm ON de.document_id = dm.document_id
    WHERE dm.processing_status = 'completed'
        AND (1 - (de.embedding <=> query_embedding)) > similarity_threshold
    ORDER BY de.embedding <=> query_embedding
    LIMIT result_limit;
END;
$$;

-- Maintenance queries for production
-- Run these periodically for optimal performance

-- Update table statistics
ANALYZE document_embeddings;
ANALYZE documents_metadata;
ANALYZE similarity_searches;

-- Vacuum for space reclamation
VACUUM (ANALYZE, VERBOSE) document_embeddings;
VACUUM (ANALYZE, VERBOSE) documents_metadata;

-- Reindex if needed (during maintenance windows)
-- REINDEX INDEX CONCURRENTLY idx_embeddings_vector_optimized;
```

---

**This deployment guide provides comprehensive production deployment strategies, security hardening, monitoring, and optimization techniques. Continue with the remaining sections in the next part.**

I've reached the character limit for this response. The deployment guide continues with sections on Backup & Recovery, CI/CD Pipeline, Scaling Strategies, and Troubleshooting Production Issues. Would you like me to complete the remaining sections in the next response?