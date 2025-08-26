#!/bin/bash

# Document Processor Deployment Script
# One-click deployment for the document processing service

set -e

echo "🚀 Starting Document Processor deployment..."

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "❌ .env file not found!"
    echo "📋 Please copy .env.example to .env and configure your environment variables:"
    echo "   cp .env.example .env"
    echo "   Edit .env with your Supabase and OpenRouter credentials"
    exit 1
fi

# Check if required environment variables are set
source .env
if [ -z "$SUPABASE_URL" ] || [ -z "$SUPABASE_ANON_KEY" ] || [ -z "$OPENROUTER_API_KEY" ]; then
    echo "❌ Missing required environment variables in .env file!"
    echo "📋 Please ensure these are set: SUPABASE_URL, SUPABASE_ANON_KEY, OPENROUTER_API_KEY"
    exit 1
fi

# Build and start the service
echo "🔨 Building Docker image..."
docker-compose build

echo "🆙 Starting services..."
docker-compose up -d

echo "⏳ Waiting for service to be healthy..."
sleep 10

# Check if service is running
if docker-compose ps | grep -q "Up"; then
    echo "✅ Document Processor is running successfully!"
    echo "🌐 Service available at: http://localhost:8000"
    echo "📊 Health check: http://localhost:8000/health"
    echo "📋 API docs: http://localhost:8000/docs"
    echo ""
    echo "📋 Useful commands:"
    echo "   View logs: docker-compose logs -f"
    echo "   Stop service: docker-compose down"
    echo "   Restart service: docker-compose restart"
else
    echo "❌ Service failed to start. Check logs with: docker-compose logs"
    exit 1
fi