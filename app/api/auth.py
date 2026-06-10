import json
from datetime import datetime, timedelta
from typing import Optional

import bcrypt
import jwt
from fastapi import APIRouter, Depends
from fastapi.security import OAuth2PasswordRequestForm

from app.db.manager import db_manager
from app.schemas.user_profile import TokenResponse, UserSignup
from src.utils.exception import AppException, ConflictError, UnauthenticatedError
from src.utils.logger import APP_LOGGER

router = APIRouter()

SECRET_KEY = "dummy_secret_key_change_in_production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 7 days

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

@router.post("/signup", response_model=dict)
async def signup(user_data: UserSignup):
    """Creates a new user account."""
    APP_LOGGER.info(f"Signup attempt for email: {user_data.email}")
    
    existing_user = await db_manager.get_user_by_email(user_data.email)
    if existing_user:
        raise ConflictError(message="Email already registered")

    try:
        # Hash password
        salt = bcrypt.gensalt()
        hashed_password = bcrypt.hashpw(user_data.password.encode('utf-8'), salt).decode('utf-8')
        
        # Serialize preferences
        prefs_json = json.dumps(user_data.preferences)
        
        # Insert user
        user_id = await db_manager.create_user(
            full_name=user_data.full_name,
            age=user_data.age,
            email=user_data.email,
            hashed_password=hashed_password,
            preferences=prefs_json
        )
        
        APP_LOGGER.info(f"Successfully created user_id: {user_id}")
        return {"message": "User created successfully", "user_id": user_id}
        
    except Exception as e:
        APP_LOGGER.error(f"Failed to create user: {e}", exc_info=True)
        raise AppException(message="Failed to create user")

@router.post("/login", response_model=TokenResponse)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """Authenticates a user and returns a JWT token."""
    APP_LOGGER.info(f"Login attempt for email: {form_data.username}")
    
    user = await db_manager.get_user_by_email(form_data.username)
    if not user:
        raise UnauthenticatedError(message="Incorrect email or password")
        
    # Verify password
    if not bcrypt.checkpw(form_data.password.encode('utf-8'), user['hashed_password'].encode('utf-8')):
        raise UnauthenticatedError(message="Incorrect email or password")
        
    # Create JWT token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": str(user['id']), "email": user['email']}, expires_delta=access_token_expires
    )
    
    APP_LOGGER.info(f"Successful login for user_id: {user['id']}")
    return TokenResponse(access_token=access_token, token_type="bearer", user_id=user['id'])
