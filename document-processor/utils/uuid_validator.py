"""
UUID validation utilities for document ID handling in the backend
"""
import re
import logging
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

class UuidValidator:
    """UUID validation utilities for document processing"""
    
    # UUID pattern for validation
    UUID_PATTERN = re.compile(
        r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
        re.IGNORECASE
    )
    
    @classmethod
    def is_valid_uuid(cls, uuid_string: Optional[str]) -> bool:
        """
        Validates if a string is a valid UUID format
        
        Args:
            uuid_string: String to validate
            
        Returns:
            bool: True if valid UUID, False otherwise
            
        Example:
            >>> UuidValidator.is_valid_uuid('c91b87e1-f9e0-438f-944c-e19217298f85')
            True
            >>> UuidValidator.is_valid_uuid('invalid-uuid')
            False
        """
        if not uuid_string or not isinstance(uuid_string, str):
            return False
        
        return bool(cls.UUID_PATTERN.match(uuid_string.strip().lower()))
    
    @classmethod
    def validate_uuid_list(cls, uuid_list: Optional[List[str]]) -> Dict[str, List[str]]:
        """
        Validates a list of UUIDs and separates valid from invalid
        
        Args:
            uuid_list: List of UUID strings to validate
            
        Returns:
            Dict with 'valid' and 'invalid' lists
            
        Example:
            >>> result = UuidValidator.validate_uuid_list(['valid-uuid-here', 'invalid'])
            >>> print(result['valid'])
            >>> print(result['invalid'])
        """
        valid = []
        invalid = []
        
        if uuid_list:
            for uuid_str in uuid_list:
                if isinstance(uuid_str, str):
                    trimmed = uuid_str.strip()
                    if trimmed:
                        if cls.is_valid_uuid(trimmed):
                            valid.append(trimmed.lower())
                        else:
                            invalid.append(uuid_str)
                else:
                    invalid.append(str(uuid_str))
        
        return {
            'valid': valid,
            'invalid': invalid
        }
    
    @classmethod
    def sanitize_uuid(cls, uuid_string: Optional[str]) -> Optional[str]:
        """
        Sanitizes a UUID string by trimming and converting to lowercase
        
        Args:
            uuid_string: UUID string to sanitize
            
        Returns:
            Sanitized UUID string or None if invalid
            
        Example:
            >>> sanitized = UuidValidator.sanitize_uuid('  C91B87E1-F9E0-438F-944C-E19217298F85  ')
            >>> print(sanitized)  # 'c91b87e1-f9e0-438f-944c-e19217298f85'
        """
        if not uuid_string or not isinstance(uuid_string, str):
            return None
        
        trimmed = uuid_string.strip().lower()
        return trimmed if cls.is_valid_uuid(trimmed) else None
    
    @classmethod
    def format_validation_log(cls, validation_result: Dict[str, List[str]]) -> str:
        """
        Formats UUID validation results for logging
        
        Args:
            validation_result: Result from validate_uuid_list()
            
        Returns:
            Formatted string for logging
        """
        valid = validation_result.get('valid', [])
        invalid = validation_result.get('invalid', [])
        
        return f"""UUID Validation Results:
  ✅ Valid ({len(valid)}): {', '.join(valid) if valid else 'None'}
  ❌ Invalid ({len(invalid)}): {', '.join(invalid) if invalid else 'None'}"""
    
    @classmethod
    def is_valid_document_id(cls, document_id: Optional[str]) -> bool:
        """
        Checks if a document ID is in the correct format for backend processing
        
        Args:
            document_id: Document ID to validate
            
        Returns:
            bool: True if valid document ID format
        """
        return cls.is_valid_uuid(document_id)
    
    @classmethod
    def get_short_id(cls, uuid_string: Optional[str]) -> str:
        """
        Extracts the first 8 characters of a UUID for display/debugging
        
        Args:
            uuid_string: UUID string
            
        Returns:
            First 8 characters or full string if shorter
            
        Example:
            >>> short = UuidValidator.get_short_id('c91b87e1-f9e0-438f-944c-e19217298f85')
            >>> print(short)  # 'c91b87e1'
        """
        if not uuid_string:
            return 'null'
        
        return uuid_string[:8] if len(uuid_string) >= 8 else uuid_string
    
    @classmethod
    def validate_and_log_document_ids(cls, document_ids: Optional[List[str]], 
                                     logger_instance: Optional[logging.Logger] = None) -> Tuple[List[str], List[str]]:
        """
        Validates document IDs and logs the results
        
        Args:
            document_ids: List of document IDs to validate
            logger_instance: Logger instance to use (defaults to module logger)
            
        Returns:
            Tuple of (valid_ids, invalid_ids)
        """
        log = logger_instance or logger
        
        if not document_ids:
            log.info("🔍 No document IDs provided for validation")
            return [], []
        
        validation = cls.validate_uuid_list(document_ids)
        valid_ids = validation['valid']
        invalid_ids = validation['invalid']
        
        log.info(f"🔍 DOCUMENT ID VALIDATION:")
        log.info(f"   Total provided: {len(document_ids)}")
        log.info(f"   Valid UUIDs: {len(valid_ids)} - {valid_ids}")
        log.info(f"   Invalid IDs: {len(invalid_ids)} - {invalid_ids}")
        
        if invalid_ids:
            log.warning(f"⚠️ Invalid document IDs detected: {invalid_ids}")
        
        return valid_ids, invalid_ids


# Convenience functions for common use cases
def validate_document_id(document_id: str) -> bool:
    """Quick validation function for single document ID"""
    return UuidValidator.is_valid_document_id(document_id)

def sanitize_document_id(document_id: str) -> Optional[str]:
    """Quick sanitization function for single document ID"""
    return UuidValidator.sanitize_uuid(document_id)

def validate_document_ids(document_ids: List[str]) -> Tuple[List[str], List[str]]:
    """Quick validation function for multiple document IDs"""
    return UuidValidator.validate_and_log_document_ids(document_ids)