import os
from supabase import create_client, Client
from dotenv import load_dotenv
import logging

# Try to import ClientOptions for compatibility
try:
    from supabase.client import ClientOptions
    HAS_CLIENT_OPTIONS = True
except ImportError:
    HAS_CLIENT_OPTIONS = False

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

def get_supabase_client() -> Client:
    """
    Get configured Supabase client
    
    Returns:
        Client: Configured Supabase client
        
    Raises:
        ValueError: If required environment variables are missing
    """
    try:
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_ANON_KEY")
        
        if not url or not key:
            raise ValueError("SUPABASE_URL and SUPABASE_ANON_KEY must be set in environment variables")
        
        # Try new API first
        try:
            client = create_client(url, key)
            logger.info("Supabase client initialized successfully")
            return client
        except TypeError as te:
            if 'proxy' in str(te):
                logger.warning(f"Supabase client initialization issue: {str(te)}")
                logger.info("Trying alternative initialization method...")
                
                # Try with ClientOptions if available
                if HAS_CLIENT_OPTIONS:
                    options = ClientOptions()
                    client = create_client(url, key, options)
                    logger.info("Supabase client initialized with ClientOptions")
                    return client
                else:
                    # Fallback: try without any additional parameters
                    client = create_client(url, key)
                    logger.info("Supabase client initialized with fallback method")
                    return client
            else:
                raise
        
    except Exception as e:
        logger.error(f"Failed to initialize Supabase client: {str(e)}")
        if 'proxy' in str(e):
            logger.error("💡 This appears to be a version compatibility issue")
            logger.error("💡 Try: pip install --upgrade supabase")
        raise

def get_supabase_service_client() -> Client:
    """
    Get Supabase client with service key for admin operations
    
    Returns:
        Client: Supabase client with service key privileges
    """
    try:
        url = os.getenv("SUPABASE_URL")
        service_key = os.getenv("SUPABASE_SERVICE_KEY")
        
        if not url or not service_key:
            raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set in environment variables")
        
        # Try new API first
        try:
            client = create_client(url, service_key)
            logger.info("Supabase service client initialized successfully")
            return client
        except TypeError as te:
            if 'proxy' in str(te):
                logger.warning(f"Supabase service client initialization issue: {str(te)}")
                logger.info("Trying alternative initialization method...")
                
                # Try with ClientOptions if available
                if HAS_CLIENT_OPTIONS:
                    options = ClientOptions()
                    client = create_client(url, service_key, options)
                    logger.info("Supabase service client initialized with ClientOptions")
                    return client
                else:
                    # Fallback: try without any additional parameters
                    client = create_client(url, service_key)
                    logger.info("Supabase service client initialized with fallback method")
                    return client
            else:
                raise
        
    except Exception as e:
        logger.error(f"Failed to initialize Supabase service client: {str(e)}")
        if 'proxy' in str(e):
            logger.error("💡 This appears to be a version compatibility issue")
            logger.error("💡 Try: pip install --upgrade supabase")
        raise

# Global client instances
_supabase_client = None
_supabase_service_client = None

def get_global_supabase_client() -> Client:
    """Get or create global Supabase client instance"""
    global _supabase_client
    if _supabase_client is None:
        _supabase_client = get_supabase_client()
    return _supabase_client

def get_global_supabase_service_client() -> Client:
    """Get or create global Supabase service client instance"""
    global _supabase_service_client
    if _supabase_service_client is None:
        _supabase_service_client = get_supabase_service_client()
    return _supabase_service_client