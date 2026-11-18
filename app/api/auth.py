import json
from datetime import timedelta

import bcrypt
from fastapi import APIRouter, Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer, OAuth2PasswordRequestForm

from app.db.manager import db_manager
from app.schemas.user_profile import RefreshTokenRequest, TokenResponse, UserSignup
from src.utils.auth_utils import (
    add_token_to_blacklist,
    create_access_token,
    create_refresh_token,
    decode_jwt,
    get_current_user,
    is_rate_limited,
)
from src.utils.exception import AppException, ConflictError, UnauthenticatedError
from src.utils.logger import APP_LOGGER

router = APIRouter()

ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 1 day
REFRESH_TOKEN_EXPIRE_DAYS = 30
_bearer_scheme = HTTPBearer(auto_error=True)


@router.post("/signup", response_model=dict)
async def signup(user_data: UserSignup) -> dict:
    """Creates a new user account."""
    APP_LOGGER.info(f"Signup attempt for email: {user_data.email}")

    existing_user = await db_manager.get_user_by_email(user_data.email)
    if existing_user:
        raise ConflictError(message="Email already registered")

    try:
        salt = bcrypt.gensalt()
        hashed_password = bcrypt.hashpw(user_data.password.encode("utf-8"), salt).decode("utf-8")
        prefs_json = json.dumps(user_data.preferences)

        user_id = await db_manager.create_user(
            full_name=user_data.full_name,
            age=user_data.age,
            email=user_data.email,
            hashed_password=hashed_password,
            preferences=prefs_json,
        )

        APP_LOGGER.info(f"Successfully created user_id: {user_id}")
        return {"message": "User created successfully", "user_id": user_id}

    except AppException:
        raise
    except Exception as e:
        APP_LOGGER.error(f"Failed to create user: {e}", exc_info=True)
        raise AppException(message="Failed to create user")  # noqa: B904


@router.post("/login", response_model=TokenResponse)
async def login(
    request: Request,
    form_data: OAuth2PasswordRequestForm = Depends(),  # noqa: B008
) -> TokenResponse:
    """Authenticates a user and returns an access + refresh token pair.
    Protected against brute-force attacks via Redis sliding-window rate limiting.
    """
    APP_LOGGER.info(f"Login attempt for email: {form_data.username}")

    client_ip = request.client.host if request.client else "unknown"
    if await is_rate_limited(client_ip):
        APP_LOGGER.warning(f"Rate limit exceeded for IP: {client_ip}")
        raise AppException(
            message="Too many login attempts. Please try again later.",
            error_code="RATE_LIMIT_EXCEEDED",
            status_code=429,
        )

    user = await db_manager.get_user_by_email(form_data.username)
    if not user:
        raise UnauthenticatedError(message="Email not registered. Please sign up.")

    if not bcrypt.checkpw(form_data.password.encode("utf-8"), user["hashed_password"].encode("utf-8")):
        raise UnauthenticatedError(message="Incorrect email or password")

    token_data = {"sub": str(user["id"]), "email": user["email"]}
    access_token = create_access_token(
        data=token_data,
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    )
    refresh_token = create_refresh_token(
        data=token_data,
        expires_delta=timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS),
    )

    APP_LOGGER.info(f"Successful login for user_id: {user['id']}")
    return TokenResponse(
        access_token=access_token,
        token_type="bearer",
        user_id=user["id"],
        refresh_token=refresh_token,
    )


@router.post("/refresh", response_model=TokenResponse)
async def refresh_access_token(body: RefreshTokenRequest) -> TokenResponse:
    """Issues a fresh access token using a valid, unrevoked refresh token."""
    payload = await decode_jwt(body.refresh_token)

    if payload.get("type") != "refresh":
        raise UnauthenticatedError(message="Invalid token type: expected refresh token")

    token_data = {"sub": payload["sub"], "email": payload["email"]}
    new_access_token = create_access_token(
        data=token_data,
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    )

    APP_LOGGER.info(f"Refreshed access token for user_id: {payload['sub']}")
    return TokenResponse(
        access_token=new_access_token,
        token_type="bearer",
        user_id=int(payload["sub"]),
    )


@router.post("/logout", response_model=dict)
async def logout(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer_scheme),  # noqa: B008
) -> dict:
    """Revokes the current access token by adding it to the Redis blacklist."""
    token = credentials.credentials
    await add_token_to_blacklist(token)
    APP_LOGGER.info("Token successfully blacklisted on logout")
    return {"message": "Successfully logged out"}
