import os
import sys
import uvicorn

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

from api.endpoints import router
from database.database import create_tables

load_dotenv()

embedding_service = None

app = FastAPI(
    title="Document Processing API",
    description="API for document upload, conversion to PDF, and OCR text extraction",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")

@app.on_event("startup")
async def startup_event():
    global embedding_service
    
    await create_tables()
    
    os.makedirs("storage/uploads", exist_ok=True)
    os.makedirs("storage/processed", exist_ok=True)
    
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_ANON_KEY")
    
    if supabase_url and supabase_key:
        try:
            from services.embedding_service import EmbeddingService
            embedding_service = EmbeddingService()
            
            import api.endpoints as endpoints_module
            endpoints_module.embedding_service = embedding_service
            
            test_embedding = embedding_service.model.encode("This is a test sentence.")
            
            
        except Exception as e:
            print(f"Embedding service initialization failed: {e}")
            embedding_service = None
            
            try:
                import api.endpoints as endpoints_module
                endpoints_module.embedding_service = None
            except:
                pass
    else:
        embedding_service = None
        
        try:
            import api.endpoints as endpoints_module
            endpoints_module.embedding_service = None
        except:
            pass

@app.get("/")
async def root():
    return {"message": "Document Processing API is running", "status": "healthy"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "document-processor",
        "version": "1.0.0"
    }

@app.get("/api/status")
async def get_status():
    supabase_status = "unknown"
    supabase_error = None
    embeddings_count = 0
    
    try:
        from supabase import create_client
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_ANON_KEY")
        
        if supabase_url and supabase_key:
            supabase = create_client(supabase_url, supabase_key)
            response = supabase.table("document_embeddings").select("*", count="exact").limit(0).execute()
            embeddings_count = response.count if hasattr(response, 'count') else 0
            supabase_status = "connected"
        else:
            supabase_status = "no_credentials"
    except Exception as e:
        supabase_status = "failed"
        supabase_error = str(e)
    
    return {
        "server": "running",
        "embedding_service": "ready" if embedding_service else "failed",
        "python_version": sys.version,
        "environment": "py311" if 'py311' in sys.executable else "unknown",
        "supabase": {
            "status": supabase_status,
            "embeddings_count": embeddings_count,
            "error": supabase_error
        },
        "environment_variables": {
            "supabase_url_set": bool(os.getenv("SUPABASE_URL")),
            "supabase_key_set": bool(os.getenv("SUPABASE_ANON_KEY")),
            "hf_model": os.getenv("HF_MODEL_NAME", "sentence-transformers/all-mpnet-base-v2")
        }
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )