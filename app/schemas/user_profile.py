from typing import Dict, Optional, Any
from pydantic import BaseModel, Field, EmailStr


class User(BaseModel):
    """
    Pydantic schema representing a User data structure.
    Used for request validation, serialization, and type safety.
    """
    user_id: int = Field(..., description="The unique identifier for the user (Primary Key)")
    full_name: str = Field(...,min_length=1,max_length=100,description="User's full name, cannot be empty",)
    age: Optional[int] = Field(None, ge=0, le=150, description="User's age, must be a positive integer")
    email: EmailStr = Field(..., max_length=255, description="Unique and validated email address")
    password: str = Field(..., min_length=8, max_length=255, description="Hashed user password string")
    preferences: Dict[str, Any] = Field(default_factory=dict,description="Arbitrary JSON structure representing user preferences",)


class UserSignup(BaseModel):
    full_name: str = Field(..., min_length=1, max_length=100)
    age: Optional[int] = Field(None, ge=0, le=150)
    email: EmailStr = Field(..., max_length=255)
    password: str = Field(..., min_length=8, max_length=255)
    preferences: Dict[str, Any] = Field(default_factory=dict)


class UserLogin(BaseModel):
    email: EmailStr = Field(..., max_length=255)
    password: str = Field(...)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: int