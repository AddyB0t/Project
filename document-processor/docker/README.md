# Docker Deployment

Quick start guide for deploying the Document Processor service using Docker.

## Prerequisites

- Docker and Docker Compose installed
- Supabase project with database setup
- OpenRouter API account

## Quick Start

1. **Configure environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env with your credentials
   ```

2. **Deploy with one command:**
   ```bash
   ./deploy.sh
   ```

## Manual Commands

If you prefer manual control:

```bash
# Build the image
docker-compose build

# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## Environment Variables

Required variables in `.env`:
- `SUPABASE_URL` - Your Supabase project URL
- `SUPABASE_ANON_KEY` - Your Supabase anonymous key  
- `OPENROUTER_API_KEY` - Your OpenRouter API key

Optional:
- `HF_MODEL_NAME` - Hugging Face model (default: sentence-transformers/all-mpnet-base-v2)

## Service URLs

Once deployed:
- API: http://localhost:8000
- Health Check: http://localhost:8000/health
- API Documentation: http://localhost:8000/docs

## Troubleshooting

- Check service status: `docker-compose ps`
- View logs: `docker-compose logs -f`
- Restart service: `docker-compose restart`
- Rebuild after code changes: `docker-compose build --no-cache`