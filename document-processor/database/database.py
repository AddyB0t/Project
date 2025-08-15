from sqlalchemy import create_engine, MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from databases import Database
import os
from typing import List, Tuple, Optional, Dict, Any
import logging

from models.document import Base, Document

logger = logging.getLogger(__name__)

DATABASE_URL = "sqlite:///./document_processor.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

database = Database(DATABASE_URL)

async def create_tables():
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Error creating database tables: {str(e)}")
        raise

def get_db():
    database_session = SessionLocal()
    try:
        yield database_session
    finally:
        database_session.close()

def create_document(db: Session, document_data: Dict[str, Any]) -> Document:
    try:
        db_document = Document(**document_data)
        db.add(db_document)
        db.commit()
        db.refresh(db_document)
        
        logger.info(f"Document created: {db_document.id}")
        return db_document
        
    except Exception as e:
        db.rollback()
        logger.error(f"Error creating document: {str(e)}")
        raise

def get_document(db: Session, document_id: str) -> Optional[Document]:
    try:
        return db.query(Document).filter(Document.id == document_id).first()
    except Exception as e:
        logger.error(f"Error getting document {document_id}: {str(e)}")
        raise

def get_all_documents(db: Session, skip: int = 0, limit: int = 50) -> Tuple[List[Document], int]:
    try:
        total = db.query(Document).count()
        
        documents = (
            db.query(Document)
            .order_by(Document.created_at.desc())
            .offset(skip)
            .limit(limit)
            .all()
        )
        
        return documents, total
        
    except Exception as e:
        logger.error(f"Error getting documents: {str(e)}")
        raise

def update_document_status(
    db: Session, 
    document_id: str, 
    status: str, 
    update_data: Optional[Dict[str, Any]] = None
) -> Optional[Document]:
    try:
        document = db.query(Document).filter(Document.id == document_id).first()
        
        if not document:
            logger.warning(f"Document not found for update: {document_id}")
            return None
        
        document.processing_status = status
        
        if update_data:
            for key, value in update_data.items():
                if hasattr(document, key):
                    setattr(document, key, value)
        
        db.commit()
        db.refresh(document)
        
        logger.info(f"Document {document_id} updated to status: {status}")
        return document
        
    except Exception as e:
        db.rollback()
        logger.error(f"Error updating document {document_id}: {str(e)}")
        raise

def delete_document(db: Session, document_id: str) -> bool:
    try:
        document = db.query(Document).filter(Document.id == document_id).first()
        
        if not document:
            return False
        
        db.delete(document)
        db.commit()
        
        logger.info(f"Document deleted: {document_id}")
        return True
        
    except Exception as e:
        db.rollback()
        logger.error(f"Error deleting document {document_id}: {str(e)}")
        raise

def get_documents_by_status(db: Session, status: str) -> List[Document]:
    try:
        return db.query(Document).filter(Document.processing_status == status).all()
    except Exception as e:
        logger.error(f"Error getting documents by status {status}: {str(e)}")
        raise

def get_documents_by_user(db: Session, user_id: str, skip: int = 0, limit: int = 50) -> Tuple[List[Document], int]:
    try:
        total = db.query(Document).filter(Document.user_id == user_id).count()
        
        documents = (
            db.query(Document)
            .filter(Document.user_id == user_id)
            .order_by(Document.created_at.desc())
            .offset(skip)
            .limit(limit)
            .all()
        )
        
        return documents, total
        
    except Exception as e:
        logger.error(f"Error getting documents for user {user_id}: {str(e)}")
        raise

def get_user_document(db: Session, document_id: str, user_id: str) -> Optional[Document]:
    try:
        return db.query(Document).filter(
            Document.id == document_id,
            Document.user_id == user_id
        ).first()
    except Exception as e:
        logger.error(f"Error getting document {document_id} for user {user_id}: {str(e)}")
        raise

def get_document_statistics(db: Session) -> Dict[str, Any]:
    try:
        total_documents = db.query(Document).count()
        
        pending = db.query(Document).filter(Document.processing_status == 'pending').count()
        processing = db.query(Document).filter(Document.processing_status == 'processing').count()
        completed = db.query(Document).filter(Document.processing_status == 'completed').count()
        failed = db.query(Document).filter(Document.processing_status == 'failed').count()
        
        completed_docs = db.query(Document).filter(
            Document.processing_status == 'completed',
            Document.processing_time.isnot(None)
        ).all()
        
        avg_processing_time = 0
        if completed_docs:
            avg_processing_time = sum(doc.processing_time for doc in completed_docs) / len(completed_docs)
        
        return {
            'total_documents': total_documents,
            'status_breakdown': {
                'pending': pending,
                'processing': processing,
                'completed': completed,
                'failed': failed
            },
            'success_rate': (completed / total_documents * 100) if total_documents > 0 else 0,
            'average_processing_time': round(avg_processing_time, 2)
        }
        
    except Exception as e:
        logger.error(f"Error getting document statistics: {str(e)}")
        raise

async def connect_database():
    try:
        await database.connect()
        logger.info("Database connected successfully")
    except Exception as e:
        logger.error(f"Error connecting to database: {str(e)}")
        raise

async def disconnect_database():
    try:
        await database.disconnect()
        logger.info("Database disconnected")
    except Exception as e:
        logger.error(f"Error disconnecting from database: {str(e)}")
        raise

