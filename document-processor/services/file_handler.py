import os
import uuid
import aiofiles
from pathlib import Path
from typing import Optional, Tuple
import magic
import logging
from fastapi import UploadFile, HTTPException

logger = logging.getLogger(__name__)

class FileHandler:
    """Handle file upload, validation, and storage operations"""
    
    def __init__(self, upload_dir: str = "storage/uploads", processed_dir: str = "storage/processed"):
        self.upload_dir = Path(upload_dir)
        self.processed_dir = Path(processed_dir)
        
        # Create directories if they don't exist
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Supported file types and size limits
        self.supported_types = {
            'application/pdf': 'pdf',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'docx',
            'application/msword': 'doc',
            'text/csv': 'csv',
            'application/csv': 'csv',
            'image/png': 'png',
            'image/jpeg': 'jpg',
            'image/jpg': 'jpg',
        }
        
        self.max_file_size = 10 * 1024 * 1024  # 10MB
    
    async def save_uploaded_file(self, file: UploadFile) -> Tuple[str, str, str]:
        """
        Save uploaded file to storage
        
        Args:
            file: FastAPI UploadFile object
            
        Returns:
            Tuple of (document_id, file_path, file_type)
        """
        try:
            # Validate file
            await self._validate_file(file)
            
            # Generate unique document ID and filename
            document_id = str(uuid.uuid4())
            file_extension = self._get_file_extension(file.filename)
            safe_filename = f"{document_id}.{file_extension}"
            
            # Full file path
            file_path = self.upload_dir / safe_filename
            
            # Save file
            await self._write_file(file, file_path)
            
            # Get file type
            file_type = self._detect_file_type(str(file_path))
            
            logger.info(f"File saved: {safe_filename} ({file.size} bytes)")
            
            return document_id, str(file_path), file_type
            
        except HTTPException:
            # Re-raise HTTPExceptions from validation without wrapping
            raise
        except Exception as e:
            logger.error(f"Error saving uploaded file: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error saving file: {str(e)}")
    
    async def _validate_file(self, file: UploadFile) -> None:
        """Validate uploaded file"""
        logger.info(f"Starting file validation for: {file.filename}")
        logger.info(f"File content type: {file.content_type}")
        logger.info(f"File size: {file.size} bytes ({file.size / (1024*1024):.2f} MB)" if file.size else "File size: None")
        logger.info(f"Max allowed size: {self.max_file_size} bytes ({self.max_file_size / (1024*1024):.1f} MB)")
        
        # Check file size
        if file.size is None:
            logger.error("File size is None")
            raise HTTPException(status_code=400, detail="File size could not be determined")
        
        if file.size > self.max_file_size:
            logger.error(f"File too large: {file.size} bytes > {self.max_file_size} bytes")
            raise HTTPException(
                status_code=413,
                detail=f"File too large ({file.size / (1024*1024):.2f} MB). Maximum size is {self.max_file_size / (1024*1024):.1f}MB"
            )
        
        logger.info("File size validation passed")
        
        # Check filename
        logger.info(f"Checking filename: '{file.filename}'")
        if not file.filename:
            logger.error("Filename is missing or empty")
            raise HTTPException(status_code=400, detail="Filename is required")
        
        logger.info("Filename validation passed")
        
        # Check file extension
        file_extension = self._get_file_extension(file.filename)
        logger.info(f"Extracted file extension: '{file_extension}'")
        if not file_extension:
            logger.error(f"No valid extension found for filename: '{file.filename}'")
            raise HTTPException(status_code=400, detail=f"File must have a valid extension. Filename: '{file.filename}'")
        
        logger.info(f"File extension validation passed: '{file_extension}'")
        logger.info("All file validations completed successfully")
    
    async def _write_file(self, file: UploadFile, file_path: Path) -> None:
        """Write uploaded file to disk"""
        try:
            async with aiofiles.open(file_path, 'wb') as f:
                content = await file.read()
                await f.write(content)
        except Exception as e:
            logger.error(f"Error writing file {file_path}: {str(e)}")
            raise HTTPException(status_code=500, detail="Error writing file to disk")
    
    def _get_file_extension(self, filename: str) -> str:
        """Extract file extension from filename"""
        return Path(filename).suffix.lower().lstrip('.')
    
    def _detect_file_type(self, file_path: str) -> str:
        """Detect file type using python-magic"""
        try:
            mime_type = magic.from_file(file_path, mime=True)
            return self.supported_types.get(mime_type, 'unknown')
        except Exception as e:
            logger.warning(f"Could not detect file type for {file_path}: {str(e)}")
            # Fallback to extension-based detection
            extension = Path(file_path).suffix.lower().lstrip('.')
            return extension if extension in ['pdf', 'docx', 'doc', 'csv', 'png', 'jpg', 'jpeg'] else 'unknown'
    
    def get_file_info(self, file_path: str) -> dict:
        """Get file information"""
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        return {
            'filename': path.name,
            'size': path.stat().st_size,
            'extension': path.suffix.lower().lstrip('.'),
            'created_at': path.stat().st_ctime,
            'modified_at': path.stat().st_mtime,
        }
    
    def cleanup_file(self, file_path: str) -> bool:
        """Delete file from storage"""
        try:
            path = Path(file_path)
            if path.exists():
                path.unlink()
                logger.info(f"File deleted: {file_path}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error deleting file {file_path}: {str(e)}")
            return False
    
    def move_to_processed(self, file_path: str) -> str:
        """Move file from uploads to processed directory"""
        try:
            source_path = Path(file_path)
            destination_path = self.processed_dir / source_path.name
            
            source_path.rename(destination_path)
            logger.info(f"File moved to processed: {destination_path}")
            
            return str(destination_path)
        except Exception as e:
            logger.error(f"Error moving file to processed: {str(e)}")
            raise