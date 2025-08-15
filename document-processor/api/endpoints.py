import asyncio
import io
import logging
import os
import time
import uuid
import requests
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import cv2
import fitz
import numpy as np
import pytesseract
from PIL import Image
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, BackgroundTasks, Form, status, Header
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from supabase import create_client

from database.database import get_db, create_document, get_document, get_all_documents, get_documents_by_user, get_user_document, update_document_status, SessionLocal
from models.document import DocumentUploadResponse, DocumentTextResponse, DocumentListResponse, ErrorResponse
from models.embedding import (
    EmbeddingRequest, EmbeddingResponse, SimilaritySearchRequest, 
    SimilaritySearchResponse, DocumentEmbeddingInfo, EmbeddingStats
)
from models.response import APIResponse, ProcessingStatusResponse
from models.user import (
    UserSignupRequest, UserLoginRequest, UserLoginResponse, UserResponse,
    TokenData, AuthStatusResponse, PasswordChangeRequest, ErrorResponse as AuthErrorResponse
)
from services.auth_service import get_auth_service, AuthService
from services.chatbot_service import ChatbotService, ChatRequest, ChatResponse, ChatMessage
from services.document_processor import DocumentProcessor
from services.embedding_service import EmbeddingService
from services.file_handler import FileHandler
from services.llm_service import get_llm_service
from services.ocr_service import OCRService

logger = logging.getLogger(__name__)

router = APIRouter()

security = HTTPBearer()
auth_service = get_auth_service()

file_handler = FileHandler()
document_processor = DocumentProcessor()
ocr_service = OCRService()
embedding_service = None
chatbot_service = ChatbotService()

_startup_validation_completed = False
_startup_validation_errors = []

def perform_startup_validation():
    global _startup_validation_completed, _startup_validation_errors
    
    if _startup_validation_completed:
        return {
            "validation_completed": True,
            "errors": _startup_validation_errors,
            "services_ready": len(_startup_validation_errors) == 0
        }
    
    logger.info("🚀 Performing startup validation...")
    startup_errors = []
    
    try:
        logger.info("🔍 Validating embedding service startup...")
        embedding_validation = validate_embedding_service()
        
        if not embedding_validation["service_available"]:
            startup_errors.extend([f"Embedding service: {error}" for error in embedding_validation["errors"]])
        else:
            logger.info("✅ Embedding service startup validation passed")
        
        logger.info("🔍 Validating LLM service startup...")
        try:
            llm_service = get_llm_service()
            if llm_service:
                logger.info("✅ LLM service startup validation passed")
            else:
                startup_errors.append("LLM service: Failed to initialize")
        except Exception as llm_error:
            startup_errors.append(f"LLM service: {str(llm_error)}")
        
        logger.info("🔍 Validating file handler startup...")
        try:
            if file_handler:
                logger.info("✅ File handler startup validation passed")
            else:
                startup_errors.append("File handler: Not initialized")
        except Exception as file_error:
            startup_errors.append(f"File handler: {str(file_error)}")
        
        logger.info("🔍 Validating document processor startup...")
        try:
            if document_processor:
                logger.info("✅ Document processor startup validation passed")
            else:
                startup_errors.append("Document processor: Not initialized")
        except Exception as doc_error:
            startup_errors.append(f"Document processor: {str(doc_error)}")
        
        logger.info("🔍 Validating chatbot service startup...")
        try:
            if chatbot_service:
                logger.info("✅ Chatbot service startup validation passed")
            else:
                startup_errors.append("Chatbot service: Not initialized")
        except Exception as chat_error:
            startup_errors.append(f"Chatbot service: {str(chat_error)}")
        
        _startup_validation_errors = startup_errors
        _startup_validation_completed = True
        
        if startup_errors:
            logger.warning(f"⚠️ Startup validation completed with {len(startup_errors)} errors")
            for error in startup_errors:
                logger.warning(f"  - {error}")
        else:
            logger.info("🎉 Startup validation completed successfully - all services ready")
        
        return {
            "validation_completed": True,
            "errors": startup_errors,
            "services_ready": len(startup_errors) == 0,
            "total_services_checked": 5,
            "validation_timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"💥 Critical error during startup validation: {str(e)}")
        startup_errors.append(f"Startup validation process: {str(e)}")
        _startup_validation_errors = startup_errors
        _startup_validation_completed = True
        
        return {
            "validation_completed": True,
            "errors": startup_errors,
            "services_ready": False,
            "critical_error": str(e)
        }

def ensure_startup_validation():
    if not _startup_validation_completed:
        return perform_startup_validation()
    
    return {
        "validation_completed": True,
        "errors": _startup_validation_errors,
        "services_ready": len(_startup_validation_errors) == 0
    }

def get_embedding_service():
    global embedding_service
    if embedding_service is None:
        try:
            logger.info("🔄 Initializing embedding service...")
            
            try:
                import sentence_transformers
                logger.info("✅ sentence-transformers dependency available")
            except ImportError as e:
                logger.error("❌ sentence-transformers not installed - embedding service cannot initialize")
                return None
            
            try:
                from supabase import create_client
                import os
                from dotenv import load_dotenv
                
                load_dotenv()
                supabase_url = os.getenv("SUPABASE_URL")
                supabase_key = os.getenv("SUPABASE_ANON_KEY")
                
                if not supabase_url or not supabase_key:
                    logger.error("❌ Supabase credentials missing - embedding service cannot initialize")
                    return None
                
                logger.info("✅ Supabase credentials available")
            except Exception as supabase_check_error:
                logger.error(f"❌ Supabase dependency check failed: {str(supabase_check_error)}")
                return None
            
            from services.embedding_service import EmbeddingService
            embedding_service = EmbeddingService()
            
            if hasattr(embedding_service, 'model') and embedding_service.model:
                logger.info("✅ Embedding service initialized successfully with model loaded")
            else:
                logger.warning("⚠️ Embedding service initialized but model may not be loaded properly")
            
            try:
                test_embedding = embedding_service.model.encode("test")
                if test_embedding is not None and len(test_embedding) > 0:
                    logger.info(f"✅ Embedding service test successful - {len(test_embedding)} dimensions")
                else:
                    logger.warning("⚠️ Embedding service test returned empty result")
            except Exception as test_error:
                logger.warning(f"⚠️ Embedding service test failed: {str(test_error)}")
            
        except ImportError as import_error:
            logger.error(f"❌ Failed to import embedding service: {str(import_error)}")
            return None
        except Exception as e:
            logger.error(f"❌ Failed to initialize embedding service: {str(e)}")
            logger.error(f"   Error type: {type(e).__name__}")
            return None
    
    return embedding_service

def get_llm_service_with_validation():
    try:
        logger.info("🔄 Validating LLM service...")
        
        import os
        from dotenv import load_dotenv
        
        load_dotenv()
        openrouter_key = os.getenv("OPENROUTER_API_KEY")
        
        if not openrouter_key:
            logger.error("❌ OPENROUTER_API_KEY not found in environment")
            return None
        
        logger.info("✅ OpenRouter API key available")
        
        llm_service = get_llm_service()
        
        if llm_service:
            try:
                api_valid = llm_service.validate_api_key()
                if api_valid:
                    logger.info("✅ LLM service validation successful")
                    return llm_service
                else:
                    logger.error("❌ LLM service API key validation failed")
                    return None
            except Exception as validation_error:
                logger.warning(f"⚠️ LLM service validation test failed: {str(validation_error)}")
                return llm_service  # Return anyway, might work for actual requests
        else:
            logger.error("❌ LLM service not available")
            return None
            
    except Exception as e:
        logger.error(f"❌ Error validating LLM service: {str(e)}")
        return None

def initialize_all_services():
    logger.info("🚀 Initializing all services...")
    
    results = {
        "embedding_service": {"initialized": False, "error": None},
        "llm_service": {"initialized": False, "error": None},
        "file_handler": {"initialized": False, "error": None},
        "document_processor": {"initialized": False, "error": None},
        "chatbot_service": {"initialized": False, "error": None},
        "total_successful": 0,
        "total_failed": 0
    }
    
    try:
        embedding_svc = get_embedding_service()
        if embedding_svc:
            results["embedding_service"]["initialized"] = True
            logger.info("✅ Embedding service initialization successful")
        else:
            results["embedding_service"]["error"] = "Service returned None"
            logger.error("❌ Embedding service initialization failed")
    except Exception as e:
        results["embedding_service"]["error"] = str(e)
        logger.error(f"❌ Embedding service initialization error: {str(e)}")
    
    try:
        llm_svc = get_llm_service_with_validation()
        if llm_svc:
            results["llm_service"]["initialized"] = True
            logger.info("✅ LLM service initialization successful")
        else:
            results["llm_service"]["error"] = "Service validation failed"
            logger.error("❌ LLM service initialization failed")
    except Exception as e:
        results["llm_service"]["error"] = str(e)
        logger.error(f"❌ LLM service initialization error: {str(e)}")
    
    try:
        if file_handler:
            results["file_handler"]["initialized"] = True
            logger.info("✅ File handler available")
        else:
            results["file_handler"]["error"] = "File handler not initialized"
    except Exception as e:
        results["file_handler"]["error"] = str(e)
    
    try:
        if document_processor:
            results["document_processor"]["initialized"] = True
            logger.info("✅ Document processor available")
        else:
            results["document_processor"]["error"] = "Document processor not initialized"
    except Exception as e:
        results["document_processor"]["error"] = str(e)
    
    try:
        if chatbot_service:
            results["chatbot_service"]["initialized"] = True
            logger.info("✅ Chatbot service available")
        else:
            results["chatbot_service"]["error"] = "Chatbot service not initialized"
    except Exception as e:
        results["chatbot_service"]["error"] = str(e)
    
    for service_name, service_result in results.items():
        if service_name not in ["total_successful", "total_failed"]:
            if service_result["initialized"]:
                results["total_successful"] += 1
            else:
                results["total_failed"] += 1
    
    logger.info(f"🎯 Service initialization complete: {results['total_successful']}/5 successful")
    
    return results

def validate_embedding_service():
    validation_result = {
        "service_available": False,
        "supabase_connection": False,
        "embedding_model_loaded": False,
        "embeddings_exist": False,
        "errors": [],
        "recommendations": [],
        "total_embeddings": 0,
        "unique_documents": 0
    }
    
    try:
        service = get_embedding_service()
        if not service:
            validation_result["errors"].append("Embedding service failed to initialize")
            validation_result["recommendations"].append("Check embedding service configuration and dependencies")
            return validation_result
        
        validation_result["service_available"] = True
        logger.info("✓ Embedding service initialized successfully")
        
        try:
            if hasattr(service, 'model') and service.model:
                validation_result["embedding_model_loaded"] = True
                logger.info("✓ Embedding model loaded successfully")
            else:
                validation_result["errors"].append("Embedding model not loaded")
                validation_result["recommendations"].append("Check sentence-transformers model loading")
        except Exception as e:
            validation_result["errors"].append(f"Error checking embedding model: {str(e)}")
        
        try:
            from supabase import create_client
            from dotenv import load_dotenv
            import os
            
            load_dotenv()
            supabase_url = os.getenv("SUPABASE_URL")
            supabase_key = os.getenv("SUPABASE_ANON_KEY")
            
            if not supabase_url or not supabase_key:
                validation_result["errors"].append("Supabase credentials not found in environment")
                validation_result["recommendations"].append("Set SUPABASE_URL and SUPABASE_ANON_KEY environment variables")
                return validation_result
            
            supabase = create_client(supabase_url, supabase_key)
            
            response = supabase.table("document_embeddings").select("document_id").limit(1).execute()
            validation_result["supabase_connection"] = True
            logger.info("✓ Supabase connection successful")
            
            count_response = supabase.table("document_embeddings").select("*", count="exact").execute()
            total_count = count_response.count if hasattr(count_response, 'count') else 0
            
            if total_count > 0:
                validation_result["embeddings_exist"] = True
                validation_result["total_embeddings"] = total_count
                
                doc_response = supabase.table("document_embeddings").select("document_id").execute()
                if doc_response.data:
                    unique_docs = set(row['document_id'] for row in doc_response.data)
                    validation_result["unique_documents"] = len(unique_docs)
                
                logger.info(f"✓ Found {total_count} embeddings from {validation_result['unique_documents']} documents")
            else:
                validation_result["errors"].append("No embeddings found in database")
                validation_result["recommendations"].append("Upload and process documents to create embeddings")
                
        except Exception as e:
            validation_result["errors"].append(f"Supabase connection error: {str(e)}")
            validation_result["recommendations"].append("Check Supabase credentials and network connectivity")
        
        if validation_result["service_available"] and validation_result["supabase_connection"] and validation_result["embeddings_exist"]:
            logger.info("✓ Embedding service validation passed - all systems operational")
        else:
            logger.warning(f"⚠ Embedding service validation issues found: {len(validation_result['errors'])} errors")
            
    except Exception as e:
        validation_result["errors"].append(f"Validation process error: {str(e)}")
        logger.error(f"Error during embedding service validation: {str(e)}")
    
    return validation_result

def get_safe_embedding_stats():
    from datetime import datetime
    
    logger.info("Getting embedding stats from document_embeddings table")
    try:
        from supabase import create_client
        from dotenv import load_dotenv
        
        load_dotenv()
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_ANON_KEY")
        
        if supabase_url and supabase_key:
            logger.info("Creating Supabase client for direct query")
            supabase = create_client(supabase_url, supabase_key)
            
            response = supabase.table("document_embeddings").select("document_id, chunk_text").execute()
            logger.info(f"Document embeddings query response: {len(response.data) if response.data else 0} records")
            
            if response.data:
                total_chunks = len(response.data)
                unique_docs = set(row['document_id'] for row in response.data)
                total_documents = len(unique_docs)
                total_characters = sum(len(row.get('chunk_text', '')) for row in response.data)
                avg_chunks = total_chunks / total_documents if total_documents > 0 else 0
                
                result = {
                    "total_documents": total_documents,
                    "total_chunks": total_chunks,
                    "total_characters": total_characters,
                    "average_chunks_per_document": round(avg_chunks, 2),
                    "embedding_dimensions": 768,
                    "last_updated": datetime.now().isoformat()
                }
                logger.info(f"Document embeddings stats: {result}")
                return result
            else:
                logger.info("No document embeddings found in database")
                return {
                    "total_documents": 0,
                    "total_chunks": 0,
                    "total_characters": 0,
                    "average_chunks_per_document": 0,
                    "embedding_dimensions": 768,
                    "last_updated": datetime.now().isoformat()
                }
        else:
            logger.error("Missing Supabase credentials")
    except Exception as e:
        logger.error(f"Error querying document_embeddings table: {e}")
    
    logger.warning("Failed to get embedding stats, returning default zeros")
    return {
        "total_documents": 0,
        "total_chunks": 0,
        "total_characters": 0,
        "average_chunks_per_document": 0,
        "embedding_dimensions": 768,
        "last_updated": datetime.now().isoformat()
    }

def safe_embedding_service_call(method_name, *args, **kwargs):
    service = get_embedding_service()
    if service and hasattr(service, method_name):
        try:
            method = getattr(service, method_name)
            return method(*args, **kwargs)
        except Exception as e:
            return None
    else:
        return None

def preprocess_for_ocr(image_part):
    open_cv_image = np.array(image_part)
    if len(open_cv_image.shape) > 2:
        open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
    
    open_cv_image = cv2.resize(open_cv_image, None, fx=2, fy=2, interpolation=cv2.INTER_LANCZOS4)
    
    gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY) if len(open_cv_image.shape) > 2 else open_cv_image
    
    denoised = cv2.medianBlur(gray, 3)
    
    binary = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    return Image.fromarray(binary)


def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    try:
        token = credentials.credentials
        token_data = auth_service.verify_token(token)
        
        if token_data is None or token_data.user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        user = auth_service.get_user_by_id(token_data.user_id)
        if user is None or not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found or inactive",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        return token_data.user_id
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in JWT middleware: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

@router.post("/register", response_model=UserResponse)
async def register_user(user_data: UserSignupRequest):
    try:
        logger.info(f"User registration attempt: {user_data.email}")
        user_response = auth_service.register_user(user_data)
        logger.info(f"User registered successfully: {user_response.user_id}")
        return user_response
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Registration error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed"
        )

@router.post("/login", response_model=UserLoginResponse)
async def login_user(login_data: UserLoginRequest):
    try:
        logger.info(f"Login attempt: {login_data.email}")
        login_response = auth_service.authenticate_user(login_data)
        logger.info(f"User logged in successfully: {login_response.user.user_id}")
        return login_response
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )

@router.get("/auth/status", response_model=AuthStatusResponse)
async def get_auth_status(current_user_id: str = Depends(get_current_user)):
    try:
        user = auth_service.get_user_by_id(current_user_id)
        if user:
            return AuthStatusResponse(
                authenticated=True,
                user=user,
                message="User is authenticated"
            )
        else:
            return AuthStatusResponse(
                authenticated=False,
                message="User not found"
            )
    except Exception as e:
        logger.error(f"Auth status error: {str(e)}")
        return AuthStatusResponse(
            authenticated=False,
            message="Authentication check failed"
        )

@router.post("/logout")
async def logout_user(current_user_id: str = Depends(get_current_user)):
    return {"message": "Successfully logged out"}

@router.post("/change-password")
async def change_password(
    password_change: PasswordChangeRequest,
    current_user_id: str = Depends(get_current_user)
):
    try:
        success = auth_service.change_password(current_user_id, password_change)
        if success:
            return {"message": "Password changed successfully"}
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to change password"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Password change error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Password change failed"
        )


@router.post("/upload-document", response_model=DocumentUploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user_id: str = Depends(get_current_user)
):
    try:
        logger.info(f"Received file upload: {file.filename} ({file.size} bytes)")
        
        document_id, file_path, file_type = await file_handler.save_uploaded_file(file)
        
        document_data = {
            'id': document_id,
            'filename': f"{document_id}.{file_type}",
            'original_filename': file.filename,
            'file_path': file_path,
            'file_size': file.size,
            'file_type': file_type,
            'processing_status': 'pending',
            'user_id': current_user_id
        }
        
        db_document = create_document(db, document_data)
        
        background_tasks.add_task(
            process_document_background,
            document_id,
            file_path,
            file.filename,
            db
        )
        
        logger.info(f"Document upload successful: {document_id}")
        
        return DocumentUploadResponse(
            document_id=document_id,
            filename=db_document.filename,
            original_filename=db_document.original_filename,
            file_size=db_document.file_size,
            file_type=db_document.file_type,
            processing_status=db_document.processing_status,
            created_at=db_document.created_at,
            user_id=db_document.user_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error uploading document: {str(e)}")

async def process_document_background(
    document_id: str,
    file_path: str,
    original_filename: str,
    db: Session
):
    try:
        logger.info(f"Starting background processing for document {document_id}")
        
        update_document_status(db, document_id, 'processing')
        
        pdf_path, extracted_text, processing_time = await document_processor.process_document(
            file_path, original_filename
        )
        
        embedding_service = get_embedding_service()
        if embedding_service and extracted_text:
            try:
                logger.info(f"Creating embeddings for document {document_id}")
                chunks_created = await embedding_service.create_embeddings(document_id, extracted_text, original_filename)
                logger.info(f"Created {chunks_created} embedding chunks for document {document_id}")
            except Exception as embed_error:
                logger.error(f"Failed to create embeddings for document {document_id}: {str(embed_error)}")
        
        update_data = {
            'pdf_path': pdf_path,
            'extracted_text': extracted_text,
            'processing_status': 'completed',
            'processing_time': processing_time
        }
        
        update_document_status(db, document_id, 'completed', update_data)
        
        logger.info(f"Document processing completed: {document_id}")
        
    except Exception as e:
        logger.error(f"Error processing document {document_id}: {str(e)}")
        
        error_data = {
            'processing_status': 'failed',
            'extracted_text': f"Processing failed: {str(e)}"
        }
        update_document_status(db, document_id, 'failed', error_data)

@router.get("/document/{document_id}/text", response_model=DocumentTextResponse)
async def get_document_text(document_id: str, db: Session = Depends(get_db), current_user_id: str = Depends(get_current_user)):
    try:
        document = get_user_document(db, document_id, current_user_id)
        
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        return DocumentTextResponse(
            document_id=document.id,
            filename=document.original_filename,
            extracted_text=document.extracted_text or "Text extraction in progress...",
            processing_status=document.processing_status,
            created_at=document.created_at,
            processing_time=document.processing_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving document text: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving document: {str(e)}")

@router.get("/document/{document_id}/status", response_model=ProcessingStatusResponse)
async def get_processing_status(document_id: str, db: Session = Depends(get_db), current_user_id: str = Depends(get_current_user)):
    try:
        document = get_user_document(db, document_id, current_user_id)
        
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        progress_map = {
            'pending': 0.0,
            'processing': 50.0,
            'completed': 100.0,
            'failed': 0.0
        }
        
        return ProcessingStatusResponse(
            document_id=document.id,
            status=document.processing_status,
            progress=progress_map.get(document.processing_status, 0.0),
            message=f"Document is {document.processing_status}",
            extracted_text=document.extracted_text if document.processing_status == 'completed' else None,
            error=document.extracted_text if document.processing_status == 'failed' else None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting processing status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting status: {str(e)}")

@router.get("/documents", response_model=DocumentListResponse)
async def list_documents(
    skip: int = 0,
    limit: int = 50,
    db: Session = Depends(get_db),
    current_user_id: str = Depends(get_current_user)
):
    try:
        documents, total = get_documents_by_user(db, current_user_id, skip=skip, limit=limit)
        
        document_responses = [
            DocumentUploadResponse(
                document_id=doc.id,
                filename=doc.filename,
                original_filename=doc.original_filename,
                file_size=doc.file_size,
                file_type=doc.file_type,
                processing_status=doc.processing_status,
                extracted_text=doc.extracted_text if doc.processing_status == 'completed' else None,
                created_at=doc.created_at,
                processing_time=doc.processing_time,
                user_id=doc.user_id
            )
            for doc in documents
        ]
        
        return DocumentListResponse(
            documents=document_responses,
            total=total
        )
        
    except Exception as e:
        logger.error(f"Error listing documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing documents: {str(e)}")

@router.delete("/document/{document_id}")
async def delete_document(document_id: str, db: Session = Depends(get_db), current_user_id: str = Depends(get_current_user)):
    try:
        document = get_user_document(db, document_id, current_user_id)
        
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        file_handler.cleanup_file(document.file_path)
        if document.pdf_path and document.pdf_path != document.file_path:
            file_handler.cleanup_file(document.pdf_path)
        
        db.delete(document)
        db.commit()
        
        return APIResponse(
            success=True,
            message="Document deleted successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}")

@router.post("/upload-and-process", response_model=DocumentUploadResponse)
async def upload_and_process_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    create_embeddings: bool = True,
    db: Session = Depends(get_db)
):
    try:
        logger.info(f"Received file upload for direct processing: {file.filename} ({file.size} bytes)")
        
        document_id, file_path, file_type = await file_handler.save_uploaded_file(file)
        
        document_data = {
            'id': document_id,
            'filename': f"{document_id}.{file_type}",
            'original_filename': file.filename,
            'file_path': file_path,
            'file_size': file.size,
            'file_type': file_type,
            'processing_status': 'pending',
            'user_id': current_user_id
        }
        
        db_document = create_document(db, document_data)
        
        background_tasks.add_task(
            process_document_directly,
            document_id,
            file_path,
            file.filename,
            create_embeddings,
            db
        )
        
        logger.info(f"Document upload with direct processing successful: {document_id}")
        
        return DocumentUploadResponse(
            document_id=document_id,
            filename=db_document.filename,
            original_filename=db_document.original_filename,
            file_size=db_document.file_size,
            file_type=db_document.file_type,
            processing_status=db_document.processing_status,
            created_at=db_document.created_at,
            user_id=db_document.user_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading document with direct processing: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error uploading document: {str(e)}")

async def process_document_directly(
    document_id: str,
    file_path: str,
    original_filename: str,
    create_embeddings: bool,
    db: Session
):
    try:
        logger.info(f"Starting direct processing workflow for document {document_id}")
        
        update_document_status(db, document_id, 'processing')
        
        pdf_path, extracted_text, processing_time = await document_processor.process_document(
            file_path, original_filename
        )
        
        
        if create_embeddings and embedding_service:
            logger.info(f"Creating embeddings for document {document_id}")
            chunks_created = await embedding_service.create_embeddings(document_id, extracted_text, original_filename)
            logger.info(f"Created {chunks_created} embedding chunks for document {document_id}")
        
        update_data = {
            'pdf_path': pdf_path,
            'extracted_text': extracted_text,
            'processing_time': processing_time,
            'processing_status': 'completed'
        }
        
        try:
            document = get_document(db, document_id)
            if document:
                for key, value in update_data.items():
                    if hasattr(document, key):
                        setattr(document, key, value)
                db.commit()
                logger.info(f"Document {document_id} processing completed successfully")
            else:
                logger.error(f"Document {document_id} not found in database")
        except Exception as db_error:
            logger.error(f"Database update error for document {document_id}: {str(db_error)}")
            db.rollback()
            
    except Exception as e:
        logger.error(f"Error in direct processing workflow for document {document_id}: {str(e)}")
        update_document_status(db, document_id, 'failed')


@router.post("/document/{document_id}/reprocess")
async def reprocess_document(
    document_id: str,
    background_tasks: BackgroundTasks,
    create_embeddings: bool = True,
    db: Session = Depends(get_db)
):
    try:
        document = get_document(db, document_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        if not os.path.exists(document.file_path):
            raise HTTPException(status_code=404, detail="Document file not found")
        
        background_tasks.add_task(
            process_document_directly,
            document_id,
            document.file_path,
            document.original_filename,
            create_embeddings,
            db
        )
        
        update_document_status(db, document_id, 'processing')
        
        return {"success": True, "message": f"Document {document_id} reprocessing started"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error reprocessing document {document_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error reprocessing document: {str(e)}")

@router.post("/ocr-image", response_model=APIResponse)
async def ocr_image_endpoint(file: UploadFile = File(...)):
    try:
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        document_id, file_path, file_type = await file_handler.save_uploaded_file(file)
        
        try:
            extracted_text = await ocr_service.extract_text_from_image(file_path)
            
            return APIResponse(
                success=True,
                message="Text extracted successfully",
                data={
                    'extracted_text': extracted_text,
                    'confidence_data': ocr_service.get_ocr_confidence(file_path)
                }
            )
            
        finally:
            file_handler.cleanup_file(file_path)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in OCR image processing: {str(e)}")
        raise HTTPException(status_code=500, detail=f"OCR processing failed: {str(e)}")




@router.post("/search-documents", response_model=SimilaritySearchResponse)
async def search_similar_documents(request: SimilaritySearchRequest):
    try:
        logger.info(f"Searching similar documents for query: {request.query[:50]}...")
        
        response = embedding_service.search_similar_documents(
            query=request.query,
            limit=request.limit,
            similarity_threshold=request.similarity_threshold
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error searching similar documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@router.get("/document/{document_id}/embeddings", response_model=DocumentEmbeddingInfo)
async def get_document_embeddings(document_id: str):
    try:
        embedding_info = embedding_service.get_document_embeddings(document_id)
        
        if not embedding_info:
            raise HTTPException(status_code=404, detail="Document embeddings not found")
        
        return embedding_info
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting document embeddings: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get embeddings: {str(e)}")

@router.delete("/document/{document_id}/embeddings")
async def delete_document_embeddings(document_id: str):
    try:
        success = embedding_service.delete_document_embeddings(document_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Document embeddings not found")
        
        return APIResponse(
            success=True,
            message=f"Embeddings deleted successfully for document {document_id}"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document embeddings: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete embeddings: {str(e)}")

@router.get("/embeddings/stats", response_model=EmbeddingStats)
async def get_embedding_statistics():
    try:
        stats = get_safe_embedding_stats()
        return stats
        
    except Exception as e:
        logger.error(f"Error getting embedding statistics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get statistics: {str(e)}")

@router.get("/debug/supabase-direct")
async def debug_supabase_direct():
    try:
        import os
        from supabase import create_client
        from dotenv import load_dotenv
        
        load_dotenv()
        
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_ANON_KEY")
        
        if not supabase_url or not supabase_key:
            return {"error": "Supabase credentials not found"}
        
        supabase = create_client(supabase_url, supabase_key)
        
        response = supabase.table("document_embeddings").select("document_id, chunk_text").limit(5).execute()
        
        count_response = supabase.table("document_embeddings").select("*", count="exact").execute()
        
        if response.data:
            unique_docs = set(row['document_id'] for row in response.data)
            return {
                "supabase_connection": "working",
                "sample_data": response.data[:2],
                "total_embeddings": count_response.count if hasattr(count_response, 'count') else len(response.data),
                "unique_documents": len(unique_docs),
                "credentials_found": True
            }
        else:
            return {
                "supabase_connection": "working",
                "sample_data": [],
                "total_embeddings": 0,
                "unique_documents": 0,
                "credentials_found": True,
                "message": "No data found in document_embeddings table"
            }
        
    except Exception as e:
        return {"error": str(e), "supabase_connection": "failed"}

@router.get("/debug/embedding-access")
async def debug_embedding_access():
    try:
        logger.info("🔍 Starting comprehensive embedding access debug...")
        
        debug_results = {
            "timestamp": time.time(),
            "startup_validation": {},
            "service_validation": {},
            "supabase_access": {},
            "embedding_model": {},
            "context_search_test": {},
            "recommendations": []
        }
        
        startup_result = ensure_startup_validation()
        debug_results["startup_validation"] = startup_result
        
        validation_result = validate_embedding_service()
        debug_results["service_validation"] = validation_result
        
        try:
            import os
            from supabase import create_client
            from dotenv import load_dotenv
            
            load_dotenv()
            supabase_url = os.getenv("SUPABASE_URL")
            supabase_key = os.getenv("SUPABASE_ANON_KEY")
            
            if supabase_url and supabase_key:
                supabase = create_client(supabase_url, supabase_key)
                
                response = supabase.table("document_embeddings").select("document_id, chunk_text, embedding").limit(3).execute()
                
                debug_results["supabase_access"] = {
                    "connection_successful": True,
                    "credentials_available": True,
                    "test_query_successful": True,
                    "sample_data_count": len(response.data) if response.data else 0,
                    "sample_has_embeddings": any('embedding' in row and row['embedding'] for row in response.data) if response.data else False
                }
            else:
                debug_results["supabase_access"] = {
                    "connection_successful": False,
                    "credentials_available": False,
                    "error": "Missing Supabase credentials"
                }
        except Exception as supabase_error:
            debug_results["supabase_access"] = {
                "connection_successful": False,
                "error": str(supabase_error)
            }
        
        try:
            service = get_embedding_service()
            if service and hasattr(service, 'model'):
                test_text = "This is a test sentence for embedding generation."
                test_embedding = service.model.encode(test_text)
                
                debug_results["embedding_model"] = {
                    "model_available": True,
                    "model_type": str(type(service.model)),
                    "test_encoding_successful": True,
                    "embedding_dimensions": len(test_embedding) if hasattr(test_embedding, '__len__') else "unknown",
                    "embedding_sample": test_embedding[:5].tolist() if hasattr(test_embedding, '__getitem__') else "not_array"
                }
            else:
                debug_results["embedding_model"] = {
                    "model_available": False,
                    "error": "Embedding model not loaded"
                }
        except Exception as model_error:
            debug_results["embedding_model"] = {
                "model_available": False,
                "error": str(model_error)
            }
        
        try:
            if validation_result["embeddings_exist"]:
                test_query = "test search query"
                search_response = chatbot_service.search_relevant_context(
                    query=test_query,
                    max_chunks=3,
                    similarity_threshold=0.1  # Low threshold for testing
                )
                
                debug_results["context_search_test"] = {
                    "search_successful": search_response.success if search_response else False,
                    "results_found": len(search_response.results) if search_response and search_response.results else 0,
                    "search_time": search_response.search_time if search_response else None,
                    "test_query": test_query
                }
            else:
                debug_results["context_search_test"] = {
                    "search_successful": False,
                    "error": "No embeddings available for testing"
                }
        except Exception as search_error:
            debug_results["context_search_test"] = {
                "search_successful": False,
                "error": str(search_error)
            }
        
        recommendations = []
        
        if not validation_result["service_available"]:
            recommendations.append("Initialize embedding service - check dependencies and configuration")
        
        if not validation_result["supabase_connection"]:
            recommendations.append("Fix Supabase connection - check credentials and network access")
        
        if not validation_result["embeddings_exist"]:
            recommendations.append("Upload and process documents to create embeddings")
        
        if not debug_results["embedding_model"].get("model_available", False):
            recommendations.append("Fix embedding model loading - check sentence-transformers installation")
        
        if not debug_results["context_search_test"].get("search_successful", False):
            recommendations.append("Fix context search functionality - check chatbot service configuration")
        
        if not recommendations:
            recommendations.append("All systems appear to be working correctly!")
        
        debug_results["recommendations"] = recommendations
        
        all_working = (
            validation_result["service_available"] and
            validation_result["supabase_connection"] and
            validation_result["embeddings_exist"] and
            debug_results["embedding_model"].get("model_available", False) and
            debug_results["context_search_test"].get("search_successful", False)
        )
        
        debug_results["overall_status"] = "healthy" if all_working else "issues_detected"
        debug_results["total_issues"] = len(recommendations) - (1 if recommendations == ["All systems appear to be working correctly!"] else 0)
        
        logger.info(f"🔍 Embedding access debug completed - Status: {debug_results['overall_status']}")
        
        return debug_results
        
    except Exception as e:
        logger.error(f"💥 Error in embedding access debug: {str(e)}")
        return {
            "error": str(e),
            "overall_status": "debug_failed",
            "timestamp": time.time()
        }

@router.post("/create-embeddings", response_model=EmbeddingResponse)
async def create_embeddings_for_text(request: EmbeddingRequest):
    try:
        document_id = str(uuid.uuid4())
        
        response = embedding_service.store_embeddings(
            document_id=document_id,
            text=request.text,
            metadata=request.metadata
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error creating embeddings for text: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create embeddings: {str(e)}")

@router.put("/document/{document_id}/embeddings", response_model=EmbeddingResponse)
async def update_document_embeddings(document_id: str, request: EmbeddingRequest):
    try:
        response = embedding_service.update_document_embeddings(
            document_id=document_id,
            text=request.text,
            metadata=request.metadata
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error updating document embeddings: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to update embeddings: {str(e)}")


@router.get("/documents/available")
async def get_available_documents():
    try:
        embedding_service = EmbeddingService()
        documents = embedding_service.get_available_document_names()
        
        logger.info(f"Retrieved {len(documents)} documents for dropdown")
        for doc in documents[:5]:  # Log first 5 for debugging
            logger.info(f"Document: {doc}")
            
        return {"success": True, "documents": documents}
    except Exception as e:
        logger.error(f"Error getting available documents: {str(e)}")
        return {"success": False, "error": str(e), "documents": []}

@router.get("/documents/{document_id}/validate")
async def validate_document(document_id: str):
    """Validate that a document exists and has embeddings available"""
    try:
        logger.info(f"🔍 Validating document: {document_id}")
        
        embedding_service = EmbeddingService()
        validation_result = embedding_service.validate_document_exists(document_id)
        
        # Add HTTP status information
        if validation_result['exists'] and validation_result['has_embeddings']:
            status_message = "Document is ready for use"
            http_status = 200
        elif validation_result['exists']:
            status_message = "Document exists but may have incomplete processing"
            http_status = 202  # Accepted but processing incomplete
        else:
            status_message = "Document not found"
            http_status = 404
        
        response_data = {
            "success": True,
            "document_id": document_id,
            "status": status_message,
            "validation": validation_result
        }
        
        logger.info(f"✅ Document validation completed for {document_id}: {status_message}")
        return JSONResponse(content=response_data, status_code=http_status)
        
    except Exception as e:
        logger.error(f"❌ Error validating document {document_id}: {str(e)}")
        return JSONResponse(
            content={
                "success": False,
                "document_id": document_id,
                "error": str(e),
                "validation": {
                    'exists': False,
                    'has_embeddings': False,
                    'errors': [str(e)]
                }
            },
            status_code=500
        )

@router.post("/documents/sync-metadata")
async def sync_existing_documents():
    """Sync existing documents from embeddings table to metadata table"""
    try:
        logger.info("🔄 Starting document metadata sync...")
        
        embedding_service = EmbeddingService()
        synced_count = embedding_service.sync_existing_documents_to_metadata()
        
        message = f"Successfully synced {synced_count} documents to metadata table"
        logger.info(f"✅ {message}")
        
        return JSONResponse(content={
            "success": True,
            "message": message,
            "synced_count": synced_count
        }, status_code=200)
        
    except Exception as e:
        error_msg = f"Error syncing document metadata: {str(e)}"
        logger.error(f"❌ {error_msg}")
        return JSONResponse(
            content={
                "success": False,
                "error": error_msg,
                "synced_count": 0
            },
            status_code=500
        )

@router.post("/documents/assign-random-uuids")
async def assign_random_uuids_to_documents():
    """Directly assign random UUIDs to the specified documents in both Supabase tables"""
    try:
        logger.info("🎲 API: Starting direct UUID assignment...")
        
        embedding_service = EmbeddingService()
        result = await embedding_service.assign_random_uuids_directly()
        
        if result['success']:
            logger.info(f"✅ API: UUID assignment completed successfully")
            return JSONResponse(content={
                "success": True,
                "message": result['message'],
                "documents_processed": result['documents_processed'],
                "successful_documents": result['successful_documents'],
                "failed_documents": result['failed_documents'],
                "total_embeddings_updated": result['total_embeddings_updated'],
                "total_metadata_updated": result['total_metadata_updated'],
                "results": result['results'],
                "successful_docs": result['successful_docs'],
                "failed_docs": result['failed_docs']
            }, status_code=200)
        else:
            logger.error(f"❌ API: UUID assignment failed")
            return JSONResponse(content={
                "success": False,
                "message": result['message'],
                "error": result.get('error', 'Unknown error'),
                "documents_processed": result.get('documents_processed', 0)
            }, status_code=500)
        
    except Exception as e:
        error_msg = f"Error in UUID assignment API: {str(e)}"
        logger.error(f"❌ API: {error_msg}")
        return JSONResponse(
            content={
                "success": False,
                "error": error_msg,
                "message": "UUID assignment API failed"
            },
            status_code=500
        )

@router.get("/documents/metadata-status")
async def check_metadata_table_status():
    """Check the status of the documents_metadata table"""
    try:
        logger.info("🔧 Checking metadata table status...")
        
        embedding_service = EmbeddingService()
        table_exists = embedding_service.ensure_metadata_table_exists()
        
        status_info = {
            "table_accessible": table_exists,
            "timestamp": datetime.now().isoformat()
        }
        
        if table_exists:
            # Get additional info about the table
            try:
                # Count records in metadata table
                metadata_count_result = embedding_service.supabase.table('documents_metadata')\
                    .select('document_id', count='exact')\
                    .execute()
                metadata_count = metadata_count_result.count if hasattr(metadata_count_result, 'count') else len(metadata_count_result.data or [])
                
                # Count records in embeddings table
                embeddings_count_result = embedding_service.supabase.table('document_embeddings')\
                    .select('document_id', count='exact')\
                    .execute()
                embeddings_count = embeddings_count_result.count if hasattr(embeddings_count_result, 'count') else len(embeddings_count_result.data or [])
                
                status_info.update({
                    "metadata_records": metadata_count,
                    "embeddings_records": embeddings_count,
                    "sync_needed": metadata_count == 0 and embeddings_count > 0
                })
                
            except Exception as count_error:
                logger.error(f"Error getting table counts: {str(count_error)}")
                status_info["count_error"] = str(count_error)
        
        return JSONResponse(content={
            "success": True,
            "status": status_info
        }, status_code=200)
        
    except Exception as e:
        error_msg = f"Error checking metadata table status: {str(e)}"
        logger.error(f"❌ {error_msg}")
        return JSONResponse(
            content={
                "success": False,
                "error": error_msg
            },
            status_code=500
        )

@router.get("/documents/sync-test")
async def test_document_sync_functionality():
    """Comprehensive test of document sync functionality"""
    try:
        logger.info("🧪 SYNC TEST: Starting comprehensive document sync test...")
        
        embedding_service = EmbeddingService()
        
        test_results = {
            "timestamp": datetime.now().isoformat(),
            "tests": {},
            "summary": {
                "passed": 0,
                "failed": 0,
                "total": 0
            }
        }
        
        # Test 1: Check metadata table accessibility
        logger.info("🧪 TEST 1: Checking metadata table accessibility...")
        try:
            table_accessible = embedding_service.ensure_metadata_table_exists()
            test_results["tests"]["metadata_table_accessible"] = {
                "status": "PASS" if table_accessible else "FAIL",
                "details": "Metadata table is accessible" if table_accessible else "Metadata table not accessible"
            }
            if table_accessible:
                test_results["summary"]["passed"] += 1
            else:
                test_results["summary"]["failed"] += 1
        except Exception as test1_error:
            test_results["tests"]["metadata_table_accessible"] = {
                "status": "FAIL",
                "details": f"Error: {str(test1_error)}"
            }
            test_results["summary"]["failed"] += 1
        test_results["summary"]["total"] += 1
        
        # Test 2: Check document availability
        logger.info("🧪 TEST 2: Checking document availability...")
        try:
            available_docs = embedding_service.get_available_document_names()
            test_results["tests"]["document_availability"] = {
                "status": "PASS" if available_docs else "FAIL",
                "details": f"Found {len(available_docs)} documents",
                "documents": [doc.get('name', 'Unknown') for doc in available_docs[:5]]  # First 5 for brevity
            }
            if available_docs:
                test_results["summary"]["passed"] += 1
            else:
                test_results["summary"]["failed"] += 1
        except Exception as test2_error:
            test_results["tests"]["document_availability"] = {
                "status": "FAIL",
                "details": f"Error: {str(test2_error)}"
            }
            test_results["summary"]["failed"] += 1
        test_results["summary"]["total"] += 1
        
        # Test 3: Test manual sync capability
        logger.info("🧪 TEST 3: Testing manual sync capability...")
        try:
            sync_count = embedding_service.sync_existing_documents_to_metadata()
            test_results["tests"]["manual_sync"] = {
                "status": "PASS",
                "details": f"Successfully synced {sync_count} documents",
                "synced_count": sync_count
            }
            test_results["summary"]["passed"] += 1
        except Exception as test3_error:
            test_results["tests"]["manual_sync"] = {
                "status": "FAIL",
                "details": f"Sync failed: {str(test3_error)}"
            }
            test_results["summary"]["failed"] += 1
        test_results["summary"]["total"] += 1
        
        # Test 4: Re-check document availability after sync
        logger.info("🧪 TEST 4: Re-checking document availability after sync...")
        try:
            post_sync_docs = embedding_service.get_available_document_names()
            test_results["tests"]["post_sync_availability"] = {
                "status": "PASS" if post_sync_docs else "FAIL",
                "details": f"Found {len(post_sync_docs)} documents after sync",
                "documents": [doc.get('name', 'Unknown') for doc in post_sync_docs[:5]]
            }
            if post_sync_docs:
                test_results["summary"]["passed"] += 1
            else:
                test_results["summary"]["failed"] += 1
        except Exception as test4_error:
            test_results["tests"]["post_sync_availability"] = {
                "status": "FAIL",
                "details": f"Error: {str(test4_error)}"
            }
            test_results["summary"]["failed"] += 1
        test_results["summary"]["total"] += 1
        
        # Overall test result
        test_results["overall_status"] = "PASS" if test_results["summary"]["failed"] == 0 else "FAIL"
        
        logger.info(f"🧪 SYNC TEST COMPLETE: {test_results['summary']['passed']}/{test_results['summary']['total']} tests passed")
        
        return JSONResponse(content={
            "success": True,
            "test_results": test_results
        }, status_code=200)
        
    except Exception as e:
        error_msg = f"Error during sync test: {str(e)}"
        logger.error(f"❌ {error_msg}")
        return JSONResponse(
            content={
                "success": False,
                "error": error_msg
            },
            status_code=500
        )

@router.get("/documents/for-selection")
async def get_documents_for_selection(db: Session = Depends(get_db)):
    try:
        documents, total = get_all_documents(db, skip=0, limit=100)
        
        embedding_stats = get_safe_embedding_stats()
        
        document_selection_list = []
        
        try:
            import os
            from supabase import create_client
            from dotenv import load_dotenv
            
            load_dotenv()
            supabase_url = os.getenv("SUPABASE_URL")
            supabase_key = os.getenv("SUPABASE_ANON_KEY")
            
            if supabase_url and supabase_key:
                supabase = create_client(supabase_url, supabase_key)
                
                for doc in documents:
                    chunk_response = supabase.table("document_embeddings").select("chunk_id, chunk_text").eq("document_id", doc.id).execute()
                    
                    chunk_count = len(chunk_response.data) if chunk_response.data else 0
                    has_embeddings = chunk_count > 0
                    
                    has_images = False
                    if doc.file_type and doc.file_type.lower() == 'pdf':
                        has_images = True  # Assume PDFs might have images
                    
                    document_info = {
                        "document_id": doc.id,
                        "filename": doc.original_filename,
                        "file_type": doc.file_type,
                        "file_size": doc.file_size,
                        "processing_status": doc.processing_status,
                        "created_at": doc.created_at.isoformat() if doc.created_at else None,
                        "chunk_count": chunk_count,
                        "has_embeddings": has_embeddings,
                        "has_images": has_images,
                        "ready_for_context": has_embeddings and doc.processing_status == 'completed',
                        "text_preview": doc.extracted_text[:200] + "..." if doc.extracted_text and len(doc.extracted_text) > 200 else doc.extracted_text or "No text extracted"
                    }
                    
                    document_selection_list.append(document_info)
                
                document_selection_list.sort(key=lambda x: (x['ready_for_context'], x['created_at'] or ''), reverse=True)
                
        except Exception as e:
            logger.warning(f"Error getting document-specific stats: {e}")
            for doc in documents:
                document_info = {
                    "document_id": doc.id,
                    "filename": doc.original_filename,
                    "file_type": doc.file_type,
                    "file_size": doc.file_size,
                    "processing_status": doc.processing_status,
                    "created_at": doc.created_at.isoformat() if doc.created_at else None,
                    "chunk_count": 0,
                    "has_embeddings": False,
                    "has_images": doc.file_type and doc.file_type.lower() == 'pdf',
                    "ready_for_context": False,
                    "text_preview": "Unable to load preview"
                }
                document_selection_list.append(document_info)
        
        return {
            "success": True,
            "total_documents": len(document_selection_list),
            "ready_documents": len([doc for doc in document_selection_list if doc['ready_for_context']]),
            "total_chunks_available": sum(doc['chunk_count'] for doc in document_selection_list),
            "documents": document_selection_list,
            "selection_capabilities": {
                "supports_multimodal": True,
                "max_documents_per_context": 10,
                "max_chunks_per_context": 15,
                "image_embedding_ready": True
            },
            "message": f"Found {len(document_selection_list)} documents, {len([doc for doc in document_selection_list if doc['ready_for_context']])} ready for context"
        }
        
    except Exception as e:
        logger.error(f"Error getting documents for selection: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get documents for selection: {str(e)}")

@router.post("/chat/prepare-context-selective")
async def prepare_chat_context_selective(
    message: str,
    selected_document_ids: List[str],
    max_chunks_per_document: int = 3,
    similarity_threshold: float = 0.3,
    include_images: bool = False
):
    try:
        logger.info(f"Preparing selective context for query: {message[:50]}... using {len(selected_document_ids)} documents")
        
        if not selected_document_ids:
            return {
                "success": False,
                "error": "No documents selected for context",
                "formatted_prompt": f"User Question: {message}",
                "context_chunks": [],
                "context_summary": "No documents selected for context",
                "message": "Please select at least one document to provide context"
            }
        
        all_context_chunks = []
        selected_sources = []
        
        service = get_embedding_service()
        if not service:
            raise HTTPException(status_code=500, detail="Embedding service not available")
        
        try:
            import os
            from supabase import create_client
            from dotenv import load_dotenv
            
            load_dotenv()
            supabase_url = os.getenv("SUPABASE_URL")
            supabase_key = os.getenv("SUPABASE_ANON_KEY")
            
            if supabase_url and supabase_key:
                supabase = create_client(supabase_url, supabase_key)
                
                query_embedding = service.model.encode(message).tolist()
                
                for doc_id in selected_document_ids:
                    doc_embeddings = supabase.table("document_embeddings").select(
                        "document_id, chunk_id, chunk_text, chunk_index, metadata, embedding"
                    ).eq("document_id", doc_id).execute()
                    
                    if doc_embeddings.data:
                        doc_similarities = []
                        
                        for row in doc_embeddings.data:
                            if 'embedding' in row and row['embedding']:
                                similarity = service._cosine_similarity(query_embedding, row['embedding'])
                                
                                if similarity >= similarity_threshold:
                                    doc_similarities.append({
                                        "document_id": row['document_id'],
                                        "chunk_id": row['chunk_id'],
                                        "chunk_text": row['chunk_text'],
                                        "chunk_index": row['chunk_index'],
                                        "similarity_score": float(similarity),
                                        "metadata": row.get('metadata', {})
                                    })
                        
                        doc_similarities.sort(key=lambda x: x['similarity_score'], reverse=True)
                        selected_chunks = doc_similarities[:max_chunks_per_document]
                        
                        all_context_chunks.extend(selected_chunks)
                        
                        if selected_chunks:
                            selected_sources.append({
                                "document_id": doc_id,
                                "chunks_used": len(selected_chunks),
                                "best_similarity": selected_chunks[0]['similarity_score'],
                                "avg_similarity": sum(chunk['similarity_score'] for chunk in selected_chunks) / len(selected_chunks)
                            })
                
                all_context_chunks.sort(key=lambda x: x['similarity_score'], reverse=True)
                final_context_chunks = all_context_chunks[:15]  # Limit to 15 total chunks
                
                if final_context_chunks:
                    context_sections = []
                    for i, chunk in enumerate(final_context_chunks, 1):
                        context_sections.append(f"""
Document Chunk {i}:
- Document ID: {chunk['document_id']}
- Similarity Score: {chunk['similarity_score']:.3f}
- Content: {chunk['chunk_text']}
---""")
                    
                    formatted_prompt = f"""You are an AI assistant that helps users understand and analyze documents. You have access to relevant document content that may help answer the user's question. Use the provided context to give accurate, helpful responses.

=== RELEVANT DOCUMENT CONTEXT ===
{chr(10).join(context_sections)}
=== END CONTEXT ===

User Question: {message}

Please provide a helpful response based on the available context:"""
                
                    context_summary = f"Used context from {len(selected_sources)} documents with {len(final_context_chunks)} relevant chunks"
                else:
                    formatted_prompt = f"""User Question: {message}

No relevant context found in the selected documents above the similarity threshold of {similarity_threshold}. Please provide a helpful response based on general knowledge:"""
                    context_summary = "No relevant context found in selected documents"
                
                return {
                    "success": True,
                    "formatted_prompt": formatted_prompt,
                    "context_chunks": final_context_chunks,
                    "context_summary": context_summary,
                    "selected_documents": selected_document_ids,
                    "sources_used": selected_sources,
                    "total_chunks_found": len(all_context_chunks),
                    "total_chunks_used": len(final_context_chunks),
                    "search_params": {
                        "similarity_threshold": similarity_threshold,
                        "max_chunks_per_document": max_chunks_per_document,
                        "include_images": include_images
                    },
                    "message": f"Context prepared from {len(selected_sources)} documents with {len(final_context_chunks)} relevant chunks"
                }
                
        except Exception as e:
            logger.error(f"Error in selective context preparation: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Selective context preparation failed: {str(e)}")
        
    except Exception as e:
        logger.error(f"Error preparing selective chat context: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to prepare selective context: {str(e)}")


@router.post("/chat/search-context")
async def search_chat_context(
    query: str = Form(...),
    max_chunks: int = Form(8),
    similarity_threshold: float = Form(0.2),
    document_name: str = Form(None)
):
    try:
        logger.info(f"Searching chat context for: {query[:50]}... (k={max_chunks})")
        
        search_response = chatbot_service.search_relevant_context(
            query=query,
            max_chunks=max_chunks,
            similarity_threshold=similarity_threshold,
            document_name=document_name
        )
        
        if not search_response.success:
            raise HTTPException(status_code=500, detail=search_response.message)
        
        response_data = {
            "success": True,
            "query": query,
            "total_results": len(search_response.results),
            "search_time": search_response.search_time,
            "context_chunks": [
                {
                    "document_id": result.document_id,
                    "chunk_id": result.chunk_id,
                    "filename": result.original_filename,
                    "content": result.chunk_text,
                    "similarity_score": result.similarity_score,
                    "chunk_index": result.chunk_index,
                    "file_type": result.file_type,
                    "metadata": result.metadata
                }
                for result in search_response.results
            ],
            "context_summary": chatbot_service.format_context_summary(search_response.results),
            "context_metadata": chatbot_service.get_context_metadata(search_response.results)
        }
        
        return JSONResponse(content=response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error searching chat context: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Context search failed: {str(e)}")




@router.post("/process-document-with-images")
async def process_document_with_images(file: UploadFile = File(...)):
    try:
        logger.info(f"Starting enhanced processing for file: {file.filename}")
        
        document_id, file_path, file_type = await file_handler.save_uploaded_file(file)
        
        package_id, package_path, extracted_text, metadata = await document_processor.process_document_with_images(
            file_path, file.filename
        )
        
        construction_analysis = analyze_construction_content(extracted_text)
        
        return {
            "success": True,
            "document_id": package_id,
            "package_path": package_path,
            "processing_summary": {
                "total_pages": metadata["total_pages"],
                "text_characters": len(extracted_text),
                "images_extracted": len(metadata["page_images"]),
                "processing_time": metadata["processing_time"]
            },
            "pages": [
                {
                    "page_number": i + 1,
                    "image_url": f"/api/document/{package_id}/page/{i + 1}/image",
                    "thumbnail_url": f"/api/document/{package_id}/thumbnail/{i + 1}",
                    "text_preview": extracted_text.split('\n\n')[i][:200] + "..." if i < len(extracted_text.split('\n\n')) else "",
                    "text_file": metadata["page_texts"][i] if i < len(metadata["page_texts"]) else ""
                }
                for i in range(metadata["total_pages"])
            ],
            "construction_analysis": construction_analysis,
            "text_preview": extracted_text[:500] + "..." if len(extracted_text) > 500 else extracted_text,
            "metadata": metadata,
            "chatbot_capabilities": {
                "can_answer_spatial_queries": True,
                "can_reference_images": True,
                "can_extract_specifications": True,
                "supports_visual_context": True
            }
        }
        
    except Exception as e:
        logger.error(f"Error in enhanced document processing: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Enhanced processing failed: {str(e)}")

@router.get("/document/{document_id}/images")
async def get_document_images(document_id: str):
    try:
        metadata = document_processor.get_document_package_info(document_id)
        
        if not metadata:
            raise HTTPException(status_code=404, detail="Document package not found")
        
        return {
            "success": True,
            "document_id": document_id,
            "total_pages": metadata["total_pages"],
            "images": [
                {
                    "page_number": i + 1,
                    "image_url": f"/api/document/{document_id}/page/{i + 1}/image",
                    "thumbnail_url": f"/api/document/{document_id}/thumbnail/{i + 1}"
                }
                for i in range(metadata["total_pages"])
            ],
            "metadata": metadata
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting document images: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get images: {str(e)}")

@router.get("/document/{document_id}/page/{page_num}/image")
async def get_page_image(document_id: str, page_num: int):
    try:
        from fastapi.responses import FileResponse
        
        image_path = Path(f"storage/document_packages/{document_id}/pages/page_{page_num}.png")
        
        if not image_path.exists():
            raise HTTPException(status_code=404, detail=f"Page {page_num} image not found")
        
        return FileResponse(
            path=str(image_path),
            media_type="image/png",
            filename=f"page_{page_num}.png"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error serving page image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to serve image: {str(e)}")

@router.get("/document/{document_id}/thumbnail/{page_num}")
async def get_page_thumbnail(document_id: str, page_num: int):
    try:
        from fastapi.responses import FileResponse
        
        thumb_path = Path(f"storage/document_packages/{document_id}/thumbnails/thumb_{page_num}.png")
        
        if not thumb_path.exists():
            raise HTTPException(status_code=404, detail=f"Page {page_num} thumbnail not found")
        
        return FileResponse(
            path=str(thumb_path),
            media_type="image/png",
            filename=f"thumb_{page_num}.png"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error serving thumbnail: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to serve thumbnail: {str(e)}")

@router.get("/document/{document_id}/package")
async def get_document_package(document_id: str):
    try:
        metadata = document_processor.get_document_package_info(document_id)
        
        if not metadata:
            raise HTTPException(status_code=404, detail="Document package not found")
        
        text_path = Path(f"storage/document_packages/{document_id}/combined_text.txt")
        combined_text = ""
        if text_path.exists():
            with open(text_path, 'r', encoding='utf-8') as f:
                combined_text = f.read()
        
        construction_analysis = analyze_construction_content(combined_text)
        
        return {
            "success": True,
            "document_id": document_id,
            "metadata": metadata,
            "text_content": combined_text,
            "construction_analysis": construction_analysis,
            "pages": [
                {
                    "page_number": i + 1,
                    "image_url": f"/api/document/{document_id}/page/{i + 1}/image",
                    "thumbnail_url": f"/api/document/{document_id}/thumbnail/{i + 1}",
                    "text_file": metadata["page_texts"][i] if i < len(metadata["page_texts"]) else ""
                }
                for i in range(metadata["total_pages"])
            ]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting document package: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get package: {str(e)}")

def analyze_construction_content(text: str) -> dict:
    import re
    
    try:
        analysis = {
            "rooms_found": [],
            "materials": [],
            "dimensions": [],
            "specifications": [],
            "drawing_types": []
        }
        
        room_patterns = [
            r'ROOM\s+\d+',
            r'OFFICE\s*\d*',
            r'RECEPTION',
            r'WAITING',
            r'GREEN\s+SPACE',
            r'PASSAGE'
        ]
        
        for pattern in room_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            analysis["rooms_found"].extend(matches)
        
        material_patterns = [
            r'EPS\s+(?:DOOR|WALL|ROOF)',
            r'Aluminum\s+windows?',
            r'\d+MM\s*\*\s*\d+MM\s*\*\s*\d+MM\s+Square\s+Tube',
            r'roofing\s+sheet',
            r'Polystyrene\s+foam',
            r'concrete',
            r'steel',
            r'block\s+wall'
        ]
        
        for pattern in material_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            analysis["materials"].extend(matches)
        
        dimension_patterns = [
            r'\d+MM?\s*[×*]\s*\d+MM?',
            r'\d+\s*[×*]\s*\d+',
            r'\d+MM'
        ]
        
        for pattern in dimension_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            analysis["dimensions"].extend(matches[:10])  # Limit to first 10
        
        spec_patterns = [
            r'EPS\s+(?:DOOR|WALL|ROOF):\s*[\d\w\s*×MM]+',
            r'Aluminum\s+windows?:\s*[\d\w\s*×MM]+',
            r'\d+mm\s+class\s+\d+\s+concrete',
            r'\d+\s+Thick\s+solid\s+block\s+wall'
        ]
        
        for pattern in spec_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            analysis["specifications"].extend(matches)
        
        if re.search(r'ROOM\s+\d+', text, re.IGNORECASE):
            analysis["drawing_types"].append("floor_plan")
        if re.search(r'elevation|front|side|rear', text, re.IGNORECASE):
            analysis["drawing_types"].append("elevations")
        if re.search(r'section|detail', text, re.IGNORECASE):
            analysis["drawing_types"].append("sections")
        if re.search(r'EPS\s+(?:DOOR|WALL|ROOF):', text, re.IGNORECASE):
            analysis["drawing_types"].append("technical_details")
        
        for key in analysis:
            if isinstance(analysis[key], list):
                analysis[key] = list(set(analysis[key]))
        
        return analysis
        
    except Exception as e:
        logger.error(f"Error analyzing construction content: {str(e)}")
        return {
            "rooms_found": [],
            "materials": [],
            "dimensions": [],
            "specifications": [],
            "drawing_types": [],
            "error": str(e)
        }


@router.post("/chat/ask-llm")
async def ask_llm_with_documents(
    user_message: str = Form(...),
    document_ids: Optional[str] = Form(None),
    max_context_chunks: int = Form(10),
    similarity_threshold: float = Form(0.2),
    max_tokens: int = Form(1000),
    temperature: float = Form(0.3)
):
    try:
        logger.info(f"🤖 LLM chat request received: '{user_message[:50]}...'")
        
        logger.info("🔍 Validating embedding service...")
        validation_result = validate_embedding_service()
        
        if not validation_result["service_available"]:
            logger.error("❌ Embedding service validation failed")
            raise HTTPException(
                status_code=500,
                detail=f"Embedding service unavailable: {'; '.join(validation_result['errors'])}"
            )
        
        if not validation_result["embeddings_exist"]:
            logger.warning("⚠️ No embeddings found in database")
            return JSONResponse(content={
                "success": True,
                "response": "I don't have access to any uploaded documents yet. Please upload and process some documents first, then I'll be able to answer questions about them.",
                "context_chunks_used": 0,
                "documents_searched": [],
                "has_context": False,
                "validation_issues": validation_result["errors"],
                "recommendations": validation_result["recommendations"],
                "message": "No embeddings available - please upload documents first"
            })
        
        logger.info(f"✅ Embedding service validated: {validation_result['total_embeddings']} embeddings from {validation_result['unique_documents']} documents")
        
        logger.info("🔍 Initializing LLM service...")
        llm_service = get_llm_service()
        if not llm_service:
            logger.error("❌ LLM service initialization failed")
            raise HTTPException(
                status_code=500, 
                detail="LLM service is not available. Please check OpenRouter API configuration."
            )
        
        logger.info("✅ LLM service ready")
        
        selected_document_ids = []
        if document_ids and document_ids.strip():
            raw_ids = [doc_id.strip() for doc_id in document_ids.split(',') if doc_id.strip()]
            
            # Use UUID validation utility
            from utils.uuid_validator import UuidValidator
            valid_ids, invalid_ids = UuidValidator.validate_and_log_document_ids(raw_ids, logger)
            
            selected_document_ids = valid_ids
            
            if valid_ids:
                logger.info(f"🎯 Document filtering enabled - searching in {len(valid_ids)} valid documents")
            else:
                logger.warning(f"⚠️ No valid document IDs provided, searching all documents")
            
            # COMPREHENSIVE DEBUG: Show what we received vs what's available
            logger.info(f"📋 DOCUMENT ID DEBUG:")
            logger.info(f"   Raw document_ids parameter: '{document_ids}'")
            logger.info(f"   Parsed document IDs: {raw_ids}")
            logger.info(f"   Valid document IDs: {valid_ids}")
            logger.info(f"   Invalid document IDs: {invalid_ids}")
            for i, doc_id in enumerate(valid_ids):
                logger.info(f"   Valid Document {i}: '{doc_id}' (length: {len(doc_id)}, type: {type(doc_id)})")
            
            # Show what documents are available in the system
            try:
                embedding_service = get_embedding_service()
                if embedding_service:
                    available_docs = embedding_service.get_available_document_names()
                    logger.info(f"📁 Available documents in system ({len(available_docs)}):")
                    for i, doc in enumerate(available_docs):
                        logger.info(f"   Available {i}: id='{doc.get('id')}', name='{doc.get('name')}'")
                    
                    # Check for exact matches
                    target_doc_id = selected_document_ids[0] if selected_document_ids else None
                    if target_doc_id:
                        matches = [doc for doc in available_docs if doc.get('id') == target_doc_id]
                        logger.info(f"🔍 Exact ID matches for '{target_doc_id}': {len(matches)}")
                        if matches:
                            logger.info(f"✅ MATCH FOUND: {matches[0]}")
                        else:
                            logger.warning(f"❌ NO EXACT MATCH for document ID: '{target_doc_id}'")
                            # Show close matches for debugging
                            close_matches = [doc for doc in available_docs if target_doc_id.lower() in doc.get('id', '').lower()]
                            logger.info(f"🔍 Close matches (case-insensitive): {close_matches}")
                            
            except Exception as debug_error:
                logger.error(f"❌ Error during document debugging: {str(debug_error)}")
        else:
            logger.info("🌐 Document filtering disabled - searching all available documents")
        
        logger.info(f"🔍 Searching for relevant context (threshold: {similarity_threshold}, max_chunks: {max_context_chunks})...")
        document_context = ""
        context_chunks_used = 0
        documents_searched = []
        context_search_errors = []
        
        try:
            search_response = None
            original_threshold = similarity_threshold
            
            # Validate and resolve document ID if provided
            selected_document_id = None
            if selected_document_ids:
                selected_document_id = selected_document_ids[0]
                logger.info(f"🎯 Document filtering enabled - target document_id: '{selected_document_id}'")
                
                # Validate that the document exists and has embeddings
                try:
                    embedding_service = get_embedding_service()
                    if embedding_service:
                        # Check if document exists in embeddings
                        check_result = embedding_service.supabase.table('document_embeddings')\
                            .select('document_id')\
                            .eq('document_id', selected_document_id)\
                            .limit(1)\
                            .execute()
                        
                        if not check_result.data:
                            logger.warning(f"⚠️ Document ID '{selected_document_id}' not found in embeddings - proceeding with all documents")
                            selected_document_id = None
                        else:
                            logger.info(f"✅ Document ID '{selected_document_id}' validated and found")
                            
                except Exception as validation_error:
                    logger.warning(f"⚠️ Document validation failed: {str(validation_error)} - proceeding with all documents")
                    selected_document_id = None
            else:
                logger.info("🌐 Document filtering disabled - searching all available documents")
            
            try:
                search_response = chatbot_service.search_relevant_context(
                    query=user_message,
                    max_chunks=max_context_chunks,
                    similarity_threshold=similarity_threshold,
                    document_id=selected_document_id
                )
                logger.info(f"✅ Context search completed with threshold {similarity_threshold}")
            except Exception as search_error:
                logger.warning(f"⚠️ Context search failed with threshold {similarity_threshold}: {str(search_error)}")
                context_search_errors.append(f"Original search failed: {str(search_error)}")
                
                fallback_threshold = max(0.1, similarity_threshold - 0.2)
                logger.info(f"🔄 Retrying with fallback threshold: {fallback_threshold}")
                
                try:
                    search_response = chatbot_service.search_relevant_context(
                        query=user_message,
                        max_chunks=max_context_chunks,
                        similarity_threshold=fallback_threshold,
                        document_id=selected_document_id
                    )
                    similarity_threshold = fallback_threshold  # Update for response
                    logger.info(f"✅ Fallback context search successful with threshold {fallback_threshold}")
                except Exception as fallback_error:
                    logger.error(f"❌ Fallback context search also failed: {str(fallback_error)}")
                    context_search_errors.append(f"Fallback search failed: {str(fallback_error)}")
            
            if search_response and search_response.success and search_response.results:
                logger.info(f"📄 Processing {len(search_response.results)} search results...")
                
                context_parts = []
                seen_documents = set()
                
                for result in search_response.results:
                    context_parts.append(f"Document: {result.original_filename}\nContent: {result.chunk_text}")
                    seen_documents.add(result.original_filename)
                    context_chunks_used += 1
                
                document_context = "\n\n---\n\n".join(context_parts)
                documents_searched = list(seen_documents)
                
                logger.info(f"✅ Context built: {context_chunks_used} chunks from {len(documents_searched)} documents")
            else:
                logger.warning("⚠️ No relevant document context found in search results")
                
                if selected_document_ids:
                    logger.info(f"🔍 Checking if selected documents exist...")
                    document_context = f"No relevant content found in the selected documents for your query. The selected documents may not contain information related to: {user_message}"
                else:
                    document_context = f"No relevant content found in any uploaded documents for your query: {user_message}"
                
        except Exception as context_error:
            logger.error(f"❌ Critical error in context search: {str(context_error)}")
            context_search_errors.append(f"Critical search error: {str(context_error)}")
            document_context = "Unable to search document context due to technical issues."
        
        if not document_context.strip():
            logger.warning("⚠️ No document context available")
            document_context = "No relevant document context found for this query. Please ensure documents are uploaded and processed."
        
        logger.info(f"📝 Context prepared: {len(document_context)} characters")
        
        logger.info("🔍 Classifying query type...")
        try:
            query_type = chatbot_service.classify_query_type(user_message)
            use_hybrid_mode = chatbot_service.should_use_hybrid_mode(user_message)
            logger.info(f"📋 Query classified as: {query_type} (hybrid_mode: {use_hybrid_mode})")
        except Exception as classify_error:
            logger.warning(f"⚠️ Query classification failed: {str(classify_error)}, defaulting to hybrid mode")
            query_type = "general"
            use_hybrid_mode = True
        
        logger.info("🤖 Calling LLM service...")
        start_time = time.time()
        
        try:
            llm_result = await llm_service.ask_question_with_context(
                user_question=user_message,
                document_context=document_context,
                max_tokens=max_tokens,
                temperature=temperature,
                use_hybrid_mode=use_hybrid_mode
            )
            processing_time = time.time() - start_time
            logger.info(f"✅ LLM response generated in {processing_time:.2f}s")
            
        except Exception as llm_error:
            processing_time = time.time() - start_time
            logger.error(f"❌ LLM service error: {str(llm_error)}")
            
            llm_result = {
                "success": False,
                "response": f"I apologize, but I encountered an error while processing your question. Context was found ({context_chunks_used} chunks from {len(documents_searched)} documents), but the AI service is currently unavailable.",
                "error": str(llm_error)
            }
        
        response_data = {
            "success": llm_result["success"],
            "response": llm_result["response"],
            "context_chunks_used": context_chunks_used,
            "documents_searched": documents_searched,
            "model_used": llm_result.get("model_used", "openai/gpt-3.5-turbo"),
            "processing_time": processing_time,
            "has_context": context_chunks_used > 0,
            "usage": llm_result.get("usage", {}),
            "query_type": query_type,
            "hybrid_mode_used": use_hybrid_mode,
            "search_params": {
                "max_context_chunks": max_context_chunks,
                "similarity_threshold": similarity_threshold,
                "original_threshold": original_threshold,
                "selected_documents": selected_document_ids,
                "temperature": temperature,
                "max_tokens": max_tokens
            },
            "validation_summary": {
                "service_available": validation_result["service_available"],
                "embeddings_available": validation_result["embeddings_exist"],
                "total_embeddings": validation_result["total_embeddings"],
                "unique_documents": validation_result["unique_documents"]
            },
            "message": llm_result.get("message", "Response generated")
        }
        
        if not llm_result["success"]:
            response_data["error"] = llm_result.get("error", "Unknown error")
        
        if context_search_errors:
            response_data["context_search_errors"] = context_search_errors
        
        if validation_result["errors"]:
            response_data["validation_warnings"] = validation_result["errors"]
        
        logger.info(f"🎉 Ask-LLM request completed successfully in {processing_time:.2f}s")
        
        return JSONResponse(content=response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"💥 Critical error in LLM chat endpoint: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to process LLM chat request: {str(e)}"
        )

@router.get("/chat/llm-status")
async def get_llm_status():
    try:
        llm_service = get_llm_service()
        
        if not llm_service:
            return {
                "success": False,
                "status": "unavailable",
                "message": "LLM service is not initialized",
                "config": {}
            }
        
        model_info = llm_service.get_model_info()
        api_key_valid = llm_service.validate_api_key()
        
        return {
            "success": True,
            "status": "available" if api_key_valid else "configuration_error",
            "message": "LLM service is ready" if api_key_valid else "API key configuration issue",
            "config": {
                **model_info,
                "document_restriction": "Only answers from uploaded documents",
                "max_tokens_default": 1000,
                "temperature_default": 0.3
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting LLM status: {str(e)}")
        return {
            "success": False,
            "status": "error",
            "message": f"Error checking LLM status: {str(e)}",
            "config": {}
        }

@router.get("/chat/context-stats")
async def get_chat_context_stats():
    try:
        logger.info("Getting chat context statistics...")
        
        stats = get_safe_embedding_stats()
        
        return {
            "success": True,
            "status": "available" if stats["total_documents"] > 0 else "no_documents",
            "total_documents": stats["total_documents"],
            "total_chunks": stats["total_chunks"],
            "total_characters": stats["total_characters"],
            "average_chunks_per_document": stats["average_chunks_per_document"],
            "embedding_dimensions": stats["embedding_dimensions"],
            "last_updated": stats["last_updated"],
            "context_available": stats["total_documents"] > 0,
            "message": f"Found {stats['total_documents']} documents with {stats['total_chunks']} chunks available for context" if stats["total_documents"] > 0 else "No documents available for context"
        }
        
    except Exception as e:
        logger.error(f"Error getting chat context statistics: {str(e)}")
        return {
            "success": False,
            "status": "error",
            "total_documents": 0,
            "total_chunks": 0,
            "total_characters": 0,
            "average_chunks_per_document": 0,
            "embedding_dimensions": 0,
            "last_updated": datetime.now(),
            "context_available": False,
            "message": f"Error getting context stats: {str(e)}"
        }

@router.get("/debug/services-status")
async def get_services_status():
    try:
        logger.info("🔍 Checking all services status...")
        
        init_results = initialize_all_services()
        
        embedding_validation = validate_embedding_service()
        startup_validation = ensure_startup_validation()
        
        llm_service_status = {
            "available": False,
            "api_key_valid": False,
            "error": None
        }
        
        try:
            llm_svc = get_llm_service_with_validation()
            if llm_svc:
                llm_service_status["available"] = True
                try:
                    llm_service_status["api_key_valid"] = llm_svc.validate_api_key()
                except:
                    llm_service_status["api_key_valid"] = False
        except Exception as llm_error:
            llm_service_status["error"] = str(llm_error)
        
        status_report = {
            "timestamp": time.time(),
            "overall_health": "unknown",
            "services": {
                "embedding_service": {
                    "status": "healthy" if embedding_validation["service_available"] else "unhealthy",
                    "details": embedding_validation
                },
                "llm_service": {
                    "status": "healthy" if llm_service_status["available"] and llm_service_status["api_key_valid"] else "unhealthy",
                    "details": llm_service_status
                },
                "file_handler": {
                    "status": "healthy" if init_results["file_handler"]["initialized"] else "unhealthy",
                    "details": init_results["file_handler"]
                },
                "document_processor": {
                    "status": "healthy" if init_results["document_processor"]["initialized"] else "unhealthy",
                    "details": init_results["document_processor"]
                },
                "chatbot_service": {
                    "status": "healthy" if init_results["chatbot_service"]["initialized"] else "unhealthy",
                    "details": init_results["chatbot_service"]
                }
            },
            "initialization_summary": init_results,
            "startup_validation": startup_validation,
            "recommendations": [],
            "critical_issues": [],
            "warnings": []
        }
        
        healthy_services = sum(1 for svc in status_report["services"].values() if svc["status"] == "healthy")
        total_services = len(status_report["services"])
        
        if healthy_services == total_services:
            status_report["overall_health"] = "healthy"
        elif healthy_services >= total_services * 0.6:  # 60% or more healthy
            status_report["overall_health"] = "degraded"
        else:
            status_report["overall_health"] = "unhealthy"
        
        if not embedding_validation["service_available"]:
            status_report["critical_issues"].append("Embedding service not available")
            status_report["recommendations"].extend(embedding_validation["recommendations"])
        
        if not embedding_validation["embeddings_exist"]:
            status_report["warnings"].append("No embeddings found in database")
            status_report["recommendations"].append("Upload and process documents to create embeddings")
        
        if not llm_service_status["available"]:
            status_report["critical_issues"].append("LLM service not available")
            status_report["recommendations"].append("Check OpenRouter API configuration")
        
        if not llm_service_status["api_key_valid"]:
            status_report["warnings"].append("LLM API key validation failed")
            status_report["recommendations"].append("Verify OpenRouter API key")
        
        if status_report["overall_health"] == "healthy":
            status_report["recommendations"].append("All services are operating normally!")
        
        logger.info(f"🎯 Services status check complete - Overall health: {status_report['overall_health']}")
        
        return status_report
        
    except Exception as e:
        logger.error(f"💥 Error checking services status: {str(e)}")
        return {
            "timestamp": time.time(),
            "overall_health": "error",
            "error": str(e),
            "critical_issues": ["Status check failed"],
            "recommendations": ["Check server logs for detailed error information"]
        }


@router.post("/documents/upload-multimodal")
async def upload_and_process_multimodal(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    extract_images: bool = Form(True),
    create_embeddings: bool = Form(True), 
    max_images: int = Form(10)
):
    try:
        logger.info(f"🤖 Multimodal upload request: {file.filename} (CLIP: {extract_images}, Embeddings: {create_embeddings})")
        
        document_id, file_path, file_type = await file_handler.save_uploaded_file(file)
        
        document_data = {
            'id': document_id,
            'filename': f"{document_id}.{file_type}",
            'original_filename': file.filename,
            'file_path': file_path,
            'file_size': file.size,
            'file_type': file_type,
            'processing_status': 'uploading'
        }
        
        db = SessionLocal()
        create_document(db, document_data)
        db.close()
        logger.info(f"Created database record for multimodal document {document_id}")
        
        background_tasks.add_task(
            process_document_multimodal_background,
            document_id=document_id,
            file_path=file_path,
            original_filename=file.filename,
            extract_images=extract_images,
            create_embeddings=create_embeddings,
            max_images=max_images
        )
        
        return {
            "success": True,
            "document_id": document_id,
            "message": "Document uploaded and queued for multimodal processing",
            "processing_status": "uploading",
            "multimodal_features": {
                "clip_analysis": extract_images,
                "text_extraction": True,
                "embedding_creation": create_embeddings,
                "max_images": max_images
            },
            "estimated_processing_time": "30-60 seconds",
            "status_check_url": f"/documents/{document_id}/status"
        }
        
    except Exception as e:
        logger.error(f"Error in multimodal upload: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Multimodal upload failed: {str(e)}")

async def process_document_multimodal_background(
    document_id: str,
    file_path: str,
    original_filename: str,
    extract_images: bool,
    create_embeddings: bool,
    max_images: int
):
    try:
        logger.info(f"🔄 Starting multimodal background processing for document {document_id}")
        
        db = SessionLocal()
        update_document_status(db, document_id, 'processing')
        
        pdf_path, extracted_text, processing_time, clip_data = await document_processor.process_document_multimodal(
            file_path, original_filename, max_images
        )
        
        if create_embeddings and embedding_service:
            logger.info(f"Creating multimodal embeddings for document {document_id}")
            
            embedding_result = await embedding_service.create_multimodal_embeddings(
                document_id=document_id,
                text=extracted_text,
                images_data=clip_data.get('images', []),
                document_name=original_filename
            )
            
            logger.info(f"Multimodal embeddings created: {embedding_result.get('total_embeddings', 0)} total")
        
        update_data = {
            'pdf_path': pdf_path,
            'extracted_text': extracted_text,
            'processing_time': processing_time,
            'processing_status': 'completed'
        }
        
        update_document_status(db, document_id, 'completed', update_data)
        
        logger.info(f"✅ Multimodal processing completed for document {document_id}")
        logger.info(f"   Text: {len(extracted_text)} chars, Images: {clip_data.get('num_images', 0)}")
        
        db.close()
        
    except Exception as e:
        logger.error(f"❌ Error in multimodal background processing for {document_id}: {str(e)}")
        
        try:
            db = SessionLocal()
            update_document_status(db, document_id, 'failed')
            db.close()
        except:
            pass

def resolve_document_identifier(document_identifier: str, embedding_service) -> str:
    try:
        if document_identifier.count('-') >= 4:
            logger.info(f"🔍 Resolving UUID to document name: {document_identifier}")
            
            try:
                available_docs = embedding_service.get_available_document_names()
                logger.info(f"DEBUG: Available documents for mapping: {available_docs}")
                
                if isinstance(available_docs, list):
                    for doc_info in available_docs:
                        if isinstance(doc_info, dict):
                            if doc_info.get('name') == document_identifier or doc_info.get('id') == document_identifier:
                                return doc_info.get('name', document_identifier)
                
                logger.warning(f"⚠️ Could not resolve UUID {document_identifier}, using as-is")
                return document_identifier
                
            except Exception as e:
                logger.warning(f"⚠️ Error resolving document identifier: {str(e)}")
                return document_identifier
        else:
            return document_identifier
            
    except Exception as e:
        logger.error(f"❌ Error in resolve_document_identifier: {str(e)}")
        return document_identifier

@router.post("/chat/ask-llm-multimodal")
async def ask_llm_multimodal(
    user_message: str = Form(...),
    document_ids: Optional[str] = Form(None),
    include_images: bool = Form(True),
    max_context_chunks: int = Form(10),
    similarity_threshold: float = Form(0.2),
    max_tokens: int = Form(1000),
    temperature: float = Form(0.3)
):
    try:
        logger.info(f"🤖 Multimodal chat request: '{user_message[:50]}...' (images: {include_images})")
        
        llm_service = get_llm_service()
        if not llm_service:
            raise HTTPException(status_code=500, detail="LLM service is not available")
        
        embedding_service = get_embedding_service()
        if not embedding_service:
            raise HTTPException(status_code=500, detail="Embedding service is not available")
        
        selected_document_ids = []
        if document_ids and document_ids.strip():
            selected_document_ids = [doc_id.strip() for doc_id in document_ids.split(',') if doc_id.strip()]
            logger.info(f"🎯 Multimodal filtering enabled - document: {selected_document_ids[0]}")
        else:
            logger.info("🌐 Multimodal filtering disabled - searching all available documents")
        
        logger.info(f"🔍 Searching multimodal context (include_images: {include_images})")
        
        try:
            available_docs = embedding_service.get_available_document_names()
            logger.info(f"DEBUG: Available documents in embeddings: {available_docs}")
        except Exception as debug_error:
            logger.warning(f"DEBUG: Could not check available documents: {str(debug_error)}")
        
        selected_document_id = selected_document_ids[0] if selected_document_ids else None
        if selected_document_id:
            logger.info(f"🎯 Using document_id for filtering: {selected_document_id}")
        
        try:
            multimodal_results = await embedding_service.search_multimodal_context(
                query=user_message,
                max_chunks=max_context_chunks,
                include_images=include_images,
                similarity_threshold=similarity_threshold,
                document_id=selected_document_id
            )
            
            context_chunks_used = multimodal_results.get('total_results', 0)
            text_chunks = multimodal_results.get('text_results', 0) 
            image_chunks = multimodal_results.get('image_results', 0)
            
            logger.info(f"📊 MULTIMODAL RESULTS: {text_chunks} text + {image_chunks} image chunks (total: {context_chunks_used})")
            
            if multimodal_results.get('diagnostics'):
                diagnostics = multimodal_results['diagnostics']
                logger.info(f"🔍 DIAGNOSTICS: Text error: {diagnostics.get('text_search_error', 'None')}, Image error: {diagnostics.get('image_search_error', 'None')}")
                if diagnostics.get('text_processing_errors'):
                    logger.warning(f"⚠️  {len(diagnostics['text_processing_errors'])} text processing errors occurred")
            
        except Exception as search_error:
            logger.warning(f"⚠️ Multimodal search failed, falling back to text-only: {str(search_error)}")
            
            text_results = chatbot_service.search_relevant_context(
                query=user_message,
                max_chunks=max_context_chunks,
                similarity_threshold=similarity_threshold,
                document_id=selected_document_id
            )
            
            multimodal_results = {
                'success': True,
                'results': text_results.results if hasattr(text_results, 'results') else [],
                'text_results': len(text_results.results) if hasattr(text_results, 'results') else 0,
                'image_results': 0
            }
            
            context_chunks_used = multimodal_results['text_results']
            text_chunks = context_chunks_used
            image_chunks = 0
        
        logger.info(f"🔧 CONTEXT BUILDING: Processing {len(multimodal_results.get('results', []))} multimodal results...")
        
        from models.embedding import SimilaritySearchResult
        
        text_results = []
        image_context_parts = []
        documents_searched = set()
        
        for result in multimodal_results.get('results', []):
            if hasattr(result, 'get'):  # Dictionary from multimodal search
                result_type = result.get('type', 'text')
                result_document_id = result.get('document_id', 'unknown')
                result_filename = result.get('original_filename', result.get('document_name', 'Unknown'))
                
                
                if result_type == 'text':
                    text_result = SimilaritySearchResult(
                        document_id=result_document_id,
                        chunk_id=result.get('chunk_id', 'unknown'),
                        chunk_text=result.get('chunk_text', ''),
                        chunk_index=result.get('chunk_index', 0),
                        similarity_score=result.get('similarity_score', 0.0),
                        original_filename=result_filename,
                        file_type=result.get('file_type', 'unknown'),
                        metadata=result.get('metadata', {})
                    )
                    text_results.append(text_result)
                    
                elif result_type == 'image':
                    metadata = result.get('metadata', {})
                    description = metadata.get('description', 'Technical drawing')
                    page_num = metadata.get('page_num', 'Unknown')
                    doc_type = result.get('document_type', 'unknown')
                    similarity_score = result.get('similarity_score', 0)
                    score_boost = result.get('score_boost', 1.0)
                    
                    image_info = f"Page {page_num} Visual ({doc_type.upper()}): {description}"
                    if similarity_score > 0.4:
                        image_info += f" [HIGH RELEVANCE: {similarity_score:.2f}]"
                    elif score_boost > 1.2:
                        image_info += f" [ENHANCED: {doc_type} optimized]"
                    
                    image_context_parts.append(image_info)
                
                documents_searched.add(result_filename)
                
            else:  # SimilaritySearchResult object from fallback
                text_results.append(result)
                documents_searched.add(result.original_filename)
        
        logger.info(f"🧠 TECHNICAL ENHANCEMENT: Processing {len(text_results)} text results, {len(image_context_parts)} image descriptions")
        
        logger.info(f"🔍 RETRIEVED CHUNKS DEBUG:")
        for i, result in enumerate(text_results):
            chunk_preview = result.chunk_text[:200] if result.chunk_text else "[EMPTY]"
            logger.info(f"  Chunk {i+1}: {chunk_preview}...")
            logger.info(f"  Similarity: {getattr(result, 'similarity_score', 'N/A')}")
            logger.info(f"  Document: {getattr(result, 'original_filename', 'Unknown')}")
            logger.info(f"  Full Length: {len(result.chunk_text) if result.chunk_text else 0} chars")
        enhanced_prompt = chatbot_service.build_context_prompt(
            user_query=user_message,
            context_results=text_results,
            conversation_history=[]
        )
        
        context_start = enhanced_prompt.find("Based on the following document information:")
        context_end = enhanced_prompt.find("=== END CONTEXT ===")
        
        if context_start >= 0 and context_end > context_start:
            document_context = enhanced_prompt[context_start:context_end + len("=== END CONTEXT ===")]
        else:
            document_context = enhanced_prompt
        
        image_context = "\n".join(image_context_parts)
        
        try:
            logger.info(f"✅ CONTEXT BUILT: {len(document_context):,} chars document context, {len(image_context):,} chars images")
        except NameError as e:
            logger.error(f"❌ CONTEXT LOGGING ERROR: {e} - Variables not defined properly")
        logger.info(f"   Text chunks: {len(text_results)}, Image descriptions: {len(image_context_parts)}, Documents: {len(documents_searched)}")
        
        query_type = chatbot_service.classify_query_type(user_message)
        use_hybrid_mode = chatbot_service.should_use_hybrid_mode(user_message)
        
        if include_images and image_chunks > 0:
            use_hybrid_mode = True
        
        try:
            logger.info(f"🤖 LLM CALL: Document context {len(document_context):,} chars, Mode: {'Hybrid' if use_hybrid_mode else 'Standard'}")
        except NameError as e:
            logger.error(f"❌ LLM CALL LOGGING ERROR: {e} - Variables not defined properly")
        logger.info(f"   Max tokens: {max_tokens}, Temperature: {temperature}, Query type: {query_type}")
        start_time = time.time()
        
        context_parts = []
        
        if multimodal_results.get('results'):
            doc_types_found = {}
            for result in multimodal_results['results']:
                doc_type = result.get('document_type', 'unknown')
                result_type = result.get('type', 'text')
                if doc_type != 'unknown':
                    if doc_type not in doc_types_found:
                        doc_types_found[doc_type] = {'text': 0, 'image': 0}
                    doc_types_found[doc_type][result_type] += 1
            
            if doc_types_found:
                type_summary = []
                for doc_type, counts in doc_types_found.items():
                    total = counts['text'] + counts['image']
                    type_summary.append(f"{doc_type.upper()}: {total} chunks ({counts['text']} text, {counts['image']} visual)")
                
                context_parts.append(f"=== DOCUMENT ANALYSIS ===")
                context_parts.append(f"Source Types: {', '.join(type_summary)}")
                context_parts.append(f"Filter: {'Document-specific search' if selected_document_id else 'All documents'}\n")
        
        context_parts.append("=== DOCUMENT CONTENT FOR ANALYSIS ===")
        context_parts.append(document_context)
        context_parts.append("=== END DOCUMENT CONTENT ===")
        
        if include_images and image_context:
            context_parts.append(f"\n=== VISUAL CONTENT ===\n{image_context}")
            
            visual_guidance = []
            for result in multimodal_results.get('results', []):
                if result.get('type') == 'image':
                    doc_type = result.get('document_type', 'unknown')
                    if doc_type == 'powerpoint':
                        visual_guidance.append("PowerPoint visuals may contain key presentation content, diagrams, or charts")
                    elif doc_type == 'excel':
                        visual_guidance.append("Excel visuals likely contain charts, graphs, or data visualizations")
                    elif doc_type == 'word':
                        visual_guidance.append("Word document visuals may include diagrams, illustrations, or embedded content")
                    elif doc_type == 'pdf':
                        visual_guidance.append("PDF visuals may contain technical drawings, charts, or formatted content")
            
            if visual_guidance:
                context_parts.append(f"\nVisual Content Guidance: {visual_guidance[0]}")
        
        enhanced_context = "\n".join(context_parts)
        logger.info(f"🔗 ENHANCED CONTEXT: Final context size {len(enhanced_context):,} chars with visual guidance")
        
        logger.info(f"🔗 CONTEXT SENT TO LLM:")
        logger.info(f"Context preview (first 500 chars): {enhanced_context[:500]}...")
        if len(enhanced_context) > 500:
            logger.info(f"Context preview (last 500 chars): ...{enhanced_context[-500:]}")
        logger.info(f"Context sections: {len(context_parts)} parts")
        
        llm_result = await llm_service.ask_question_with_context(
            user_question=user_message,
            document_context=enhanced_context,
            max_tokens=max_tokens,
            temperature=temperature,
            use_hybrid_mode=use_hybrid_mode
        )
        
        processing_time = time.time() - start_time
        
        response_data = {
            "success": llm_result["success"],
            "response": llm_result["response"],
            "context_chunks_used": context_chunks_used,
            "text_chunks": text_chunks,
            "image_chunks": image_chunks,
            "documents_searched": list(documents_searched),
            "model_used": llm_result.get("model_used", "openai/gpt-3.5-turbo"),
            "processing_time": processing_time,
            "has_multimodal_context": image_chunks > 0,
            "multimodal_mode": include_images,
            "query_type": query_type,
            "hybrid_mode_used": use_hybrid_mode,
            "usage": llm_result.get("usage", {}),
            "search_diagnostics": multimodal_results.get('diagnostics', {}),
            "search_params": {
                "max_context_chunks": max_context_chunks,
                "similarity_threshold": similarity_threshold,
                "include_images": include_images,
                "selected_documents": selected_document_ids,
                "temperature": temperature,
                "max_tokens": max_tokens
            },
            "message": f"Multimodal response generated using {text_chunks} text + {image_chunks} image chunks"
        }
        
        logger.info(f"✅ MULTIMODAL COMPLETE: {processing_time:.2f}s total, LLM: {llm_result.get('processing_time', 0):.2f}s")
        logger.info(f"   Response length: {len(response_data.get('response', '')):,} chars, Model: {response_data.get('model_used', 'unknown')}")
        
        logger.info(f"📊 SESSION SUMMARY: {context_chunks_used} chunks → {len(response_data.get('response', '')):,} chars ({processing_time:.2f}s)")
        
        return JSONResponse(content=response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"🚨 MULTIMODAL ENDPOINT ERROR: {str(e)}", exc_info=True)
        logger.error(f"   Query: '{user_message[:100]}{'...' if len(user_message) > 100 else ''}'")
        logger.error(f"   Selected document: {selected_document_ids}")
        raise HTTPException(
            status_code=500, 
            detail={
                "error": str(e),
                "error_type": type(e).__name__,
                "debug_info": {
                    "query_length": len(user_message),
                    "selected_document_ids": selected_document_ids,
                    "max_context_chunks": max_context_chunks,
                    "include_images": include_images
                }
            }
        )