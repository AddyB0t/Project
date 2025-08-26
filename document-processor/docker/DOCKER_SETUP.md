# Docker Setup Guide for Document Processor

This guide will help you deploy the Document Processor backend using Docker, eliminating all environment setup issues including the tokenizers build failure.

## Prerequisites

1. **Install Docker** (if not already installed):
   ```bash
   # Ubuntu/Debian
   sudo apt update
   sudo apt install docker.io docker-compose
   sudo systemctl start docker
   sudo systemctl enable docker
   sudo usermod -aG docker $USER
   # Log out and back in for group changes
   
   # Windows/Mac: Download Docker Desktop from https://www.docker.com/products/docker-desktop
   ```

2. **Verify Docker Installation**:
   ```bash
   docker --version
   docker-compose --version
   ```

## Quick Start (One-Command Deployment)

1. **Configure Environment**:
   ```bash
   # Copy environment template
   cp .env.example .env
   
   # Edit .env with your credentials
   nano .env  # or use any text editor
   ```

2. **Deploy with Docker Compose**:
   ```bash
   docker-compose up --build -d
   ```

3. **Check Status**:
   ```bash
   # View logs
   docker-compose logs -f
   
   # Check health
   curl http://localhost:8000/health
   ```

## Manual Build (Alternative)

If you prefer to build and run manually:

```bash
# Build image
docker build -t document-processor .

# Run container
docker run -d \
  --name document-processor \
  -p 8000:8000 \
  --env-file .env \
  -v $(pwd)/storage:/app/storage \
  -v $(pwd)/document_processor.db:/app/document_processor.db \
  document-processor
```

## Environment Variables

Create `.env` file with these variables:

```env
# Required - Get from Supabase Dashboard
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=your-supabase-anon-key

# Optional - For LLM features
OPENROUTER_API_KEY=your-openrouter-api-key

# Optional - Default model
HF_MODEL_NAME=sentence-transformers/all-mpnet-base-v2
```

## Useful Commands

```bash
# View container logs
docker-compose logs -f document-processor

# Stop services
docker-compose down

# Rebuild after code changes
docker-compose up --build -d

# Access container shell
docker-compose exec document-processor bash

# Check container resource usage
docker stats
```

## Troubleshooting

- **Port 8000 already in use**: Change port in docker-compose.yml from "8000:8000" to "8001:8000"
- **Permission errors**: Ensure your user is in docker group: `sudo usermod -aG docker $USER`
- **Build failures**: Clear Docker cache: `docker system prune -a`

## Benefits

✅ **No Environment Issues**: Pre-built Python 3.11 with all dependencies
✅ **No Tokenizers Build Errors**: Build tools included in container
✅ **Consistent Deployment**: Same environment across all machines
✅ **Easy Updates**: Rebuild container for updates
✅ **Port Mapping**: Access backend at http://localhost:8000