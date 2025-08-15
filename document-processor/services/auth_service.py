import bcrypt
import jwt
import uuid
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from fastapi import HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from config.supabase_config import get_global_supabase_service_client
from database.database import get_db
from models.user import (
    User, UserSignupRequest, UserLoginRequest, UserResponse, 
    UserLoginResponse, TokenData, TokenResponse, PasswordChangeRequest
)

logger = logging.getLogger(__name__)

class AuthService:
    def __init__(self, secret_key: str = "your-secret-key", algorithm: str = "HS256", access_token_expire_minutes: int = 30):
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.access_token_expire_minutes = access_token_expire_minutes
        self.supabase = get_global_supabase_service_client()

    def _hash_password(self, password: str) -> str:
        password_bytes = password.encode('utf-8')
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password_bytes, salt)
        return hashed.decode('utf-8')

    def _verify_password(self, plain_password: str, hashed_password: str) -> bool:
        password_bytes = plain_password.encode('utf-8')
        hashed_bytes = hashed_password.encode('utf-8')
        return bcrypt.checkpw(password_bytes, hashed_bytes)

    def _get_user_from_local_db(self, email: str) -> Optional[Dict[str, Any]]:
        try:
            db_gen = get_db()
            db = next(db_gen)
            try:
                user = db.query(User).filter(User.email == email).first()
                if user:
                    return {
                        "user_id": user.user_id,
                        "email": user.email,
                        "password_hash": user.password_hash,
                        "first_name": user.first_name,
                        "last_name": user.last_name,
                        "is_active": user.is_active,
                        "created_at": user.created_at.isoformat() if user.created_at else None,
                        "updated_at": user.updated_at.isoformat() if user.updated_at else None
                    }
                return None
            finally:
                db.close()
        except Exception as e:
            logger.error(f"Error accessing local database: {str(e)}")
            return None

    def create_access_token(self, data: dict, expires_delta: Optional[timedelta] = None) -> str:
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)
        
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt

    def verify_token(self, token: str) -> Optional[TokenData]:
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            user_id: str = payload.get("sub")
            email: str = payload.get("email")
            if user_id is None:
                return None
            return TokenData(user_id=user_id, email=email)
        except jwt.ExpiredSignatureError:
            logger.warning("Token has expired")
            return None
        except jwt.InvalidTokenError:
            logger.warning("Invalid token")
            return None

    def register_user(self, user_data: UserSignupRequest) -> UserResponse:
        try:
            existing_user = self.supabase.table("users").select("*").eq("email", user_data.email).execute()
            if existing_user.data:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Email already registered"
                )

            user_id = str(uuid.uuid4())
            
            password_hash = self._hash_password(user_data.password)
            
            user_insert_data = {
                "user_id": user_id,
                "email": user_data.email,
                "password_hash": password_hash,
                "first_name": user_data.first_name,
                "last_name": user_data.last_name,
                "is_active": True
            }
            
            result = self.supabase.table("users").insert(user_insert_data).execute()
            
            if not result.data:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to create user"
                )
            
            created_user = result.data[0]
            
            return UserResponse(
                user_id=created_user["user_id"],
                email=created_user["email"],
                first_name=created_user.get("first_name"),
                last_name=created_user.get("last_name"),
                is_active=created_user["is_active"],
                created_at=datetime.fromisoformat(created_user["created_at"].replace('Z', '+00:00')),
                updated_at=datetime.fromisoformat(created_user["updated_at"].replace('Z', '+00:00')) if created_user.get("updated_at") else None
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error registering user: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal server error during registration"
            )

    def authenticate_user(self, login_data: UserLoginRequest) -> UserLoginResponse:
        """Authenticate user and return user data with access token"""
        try:
            # First try to get user from Supabase
            user_result = self.supabase.table("users").select("*").eq("email", login_data.email).execute()
            user_data = None
            
            if user_result.data:
                user_data = user_result.data[0]
                logger.info(f"User found in Supabase: {login_data.email}")
            else:
                # If not found in Supabase, try local database
                logger.info(f"User not found in Supabase, checking local database: {login_data.email}")
                user_data = self._get_user_from_local_db(login_data.email)
                
                if user_data:
                    logger.info(f"User found in local database: {login_data.email}")
                
            if not user_data:
                logger.warning(f"User not found in either database: {login_data.email}")
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid email or password"
                )
            
            # Check if user is active
            if not user_data.get("is_active", True):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Account is deactivated"
                )
            
            # Verify password
            if not self._verify_password(login_data.password, user_data["password_hash"]):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid email or password"
                )
            
            # Create access token
            access_token_expires = timedelta(minutes=self.access_token_expire_minutes)
            access_token_data = {
                "sub": user_data["user_id"],
                "email": user_data["email"]
            }
            access_token = self.create_access_token(
                data=access_token_data, 
                expires_delta=access_token_expires
            )
            
            # Prepare user response - handle different timestamp formats
            created_at = user_data.get("created_at")
            if isinstance(created_at, str):
                if 'Z' in created_at:
                    # Supabase format
                    created_at = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                else:
                    # Local database format
                    created_at = datetime.fromisoformat(created_at)
            elif created_at is None:
                created_at = datetime.utcnow()
            
            updated_at = user_data.get("updated_at")
            if isinstance(updated_at, str):
                if 'Z' in updated_at:
                    # Supabase format
                    updated_at = datetime.fromisoformat(updated_at.replace('Z', '+00:00'))
                else:
                    # Local database format
                    updated_at = datetime.fromisoformat(updated_at)
            else:
                updated_at = None

            user_response = UserResponse(
                user_id=user_data["user_id"],
                email=user_data["email"],
                first_name=user_data.get("first_name"),
                last_name=user_data.get("last_name"),
                is_active=user_data["is_active"],
                created_at=created_at,
                updated_at=updated_at
            )
            
            return UserLoginResponse(
                user=user_response,
                access_token=access_token,
                token_type="bearer",
                expires_in=self.access_token_expire_minutes * 60  # Convert to seconds
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error authenticating user: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal server error during authentication"
            )

    def get_user_by_id(self, user_id: str) -> Optional[UserResponse]:
        """Get user by user_id - check both Supabase and local database"""
        try:
            # First try Supabase
            user_result = self.supabase.table("users").select("*").eq("user_id", user_id).execute()
            user_data = None
            
            if user_result.data:
                user_data = user_result.data[0]
                logger.info(f"User {user_id} found in Supabase")
            else:
                # Try local database
                logger.info(f"User {user_id} not found in Supabase, checking local database")
                db_gen = get_db()
                db = next(db_gen)
                try:
                    local_user = db.query(User).filter(User.user_id == user_id).first()
                    if local_user:
                        logger.info(f"User {user_id} found in local database")
                        user_data = {
                            "user_id": local_user.user_id,
                            "email": local_user.email,
                            "first_name": local_user.first_name,
                            "last_name": local_user.last_name,
                            "is_active": local_user.is_active,
                            "created_at": local_user.created_at.isoformat() if local_user.created_at else None,
                            "updated_at": local_user.updated_at.isoformat() if local_user.updated_at else None
                        }
                finally:
                    db.close()
            
            if not user_data:
                logger.warning(f"User {user_id} not found in either database")
                return None
            
            # Handle different timestamp formats
            created_at = user_data.get("created_at")
            if isinstance(created_at, str):
                if 'Z' in created_at:
                    # Supabase format
                    created_at = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                else:
                    # Local database format
                    created_at = datetime.fromisoformat(created_at)
            elif created_at is None:
                created_at = datetime.utcnow()
            
            updated_at = user_data.get("updated_at")
            if isinstance(updated_at, str):
                if 'Z' in updated_at:
                    # Supabase format
                    updated_at = datetime.fromisoformat(updated_at.replace('Z', '+00:00'))
                else:
                    # Local database format
                    updated_at = datetime.fromisoformat(updated_at)
            else:
                updated_at = None
            
            return UserResponse(
                user_id=user_data["user_id"],
                email=user_data["email"],
                first_name=user_data.get("first_name"),
                last_name=user_data.get("last_name"),
                is_active=user_data["is_active"],
                created_at=created_at,
                updated_at=updated_at
            )
            
        except Exception as e:
            logger.error(f"Error getting user by ID: {str(e)}")
            return None

    def change_password(self, user_id: str, password_change: PasswordChangeRequest) -> bool:
        """Change user password"""
        try:
            # Get current user data
            user_result = self.supabase.table("users").select("*").eq("user_id", user_id).execute()
            
            if not user_result.data:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="User not found"
                )
            
            user_data = user_result.data[0]
            
            # Verify current password
            if not self._verify_password(password_change.current_password, user_data["password_hash"]):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid current password"
                )
            
            # Hash new password
            new_password_hash = self._hash_password(password_change.new_password)
            
            # Update password in database
            update_result = self.supabase.table("users").update({
                "password_hash": new_password_hash,
                "updated_at": datetime.utcnow().isoformat()
            }).eq("user_id", user_id).execute()
            
            return bool(update_result.data)
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error changing password: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal server error during password change"
            )

    def deactivate_user(self, user_id: str) -> bool:
        """Deactivate a user account"""
        try:
            update_result = self.supabase.table("users").update({
                "is_active": False,
                "updated_at": datetime.utcnow().isoformat()
            }).eq("user_id", user_id).execute()
            
            return bool(update_result.data)
            
        except Exception as e:
            logger.error(f"Error deactivating user: {str(e)}")
            return False

# Global instance
_auth_service = None

def get_auth_service() -> AuthService:
    """Get or create global AuthService instance"""
    global _auth_service
    if _auth_service is None:
        # In production, these should come from environment variables
        _auth_service = AuthService(
            secret_key="your-super-secret-key-change-in-production",
            algorithm="HS256",
            access_token_expire_minutes=60  # 1 hour
        )
    return _auth_service