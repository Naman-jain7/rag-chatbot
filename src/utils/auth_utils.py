from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
import jwt
import redis.asyncio as aioredis

from configs.core import settings
from src.utils.exception import UnauthenticatedError
from src.utils.logger import APP_LOGGER

# ---------------------------------------------------------------------------
# Redis client (async singleton with graceful fallback)
# ---------------------------------------------------------------------------

_redis: aioredis.Redis | None = None

# In-memory fallbacks when Redis is disabled or unreachable
_MEMORY_BLACKLIST: set[str] = set()
_MEMORY_RATE_LIMIT: dict[str, list[float]] = {}


def get_redis() -> aioredis.Redis | None:
    """Return (or lazily create) the shared async Redis client.
    Returns None if Redis is disabled in settings.
    """
    global _redis
    if not getattr(settings.redis, "REDIS_ENABLED", True):
        return None

    if _redis is None:
        try:
            _redis = aioredis.Redis(
                host=settings.redis.REDIS_HOST,
                port=settings.redis.REDIS_PORT,
                db=settings.redis.REDIS_DB,
                password=settings.redis.REDIS_PASSWORD,
                socket_timeout=settings.redis.REDIS_SOCKET_TIMEOUT,
                decode_responses=True,
            )
        except Exception as e:
            APP_LOGGER.error(f"Failed to initialize Redis client: {e}", exc_info=True)
            return None

    return _redis


async def close_redis() -> None:
    """Close the Redis client connection pool."""
    global _redis
    if _redis is not None:
        try:
            await _redis.aclose()
        except Exception as e:
            APP_LOGGER.error(f"Error closing Redis client: {e}", exc_info=True)
        finally:
            _redis = None


# ---------------------------------------------------------------------------
# Key prefixes and defaults
# ---------------------------------------------------------------------------

_BLACKLIST_PREFIX = "auth:blacklist:"
_RATE_LIMIT_PREFIX = "auth:rate:"

# Default fallback TTL: 7 days
_DEFAULT_BLACKLIST_TTL_SECONDS = 60 * 60 * 24 * 7

# Rate-limit window and maximum attempts
_DEFAULT_RATE_WINDOW_SECONDS = 60
_DEFAULT_RATE_MAX_ATTEMPTS = 5


# ---------------------------------------------------------------------------
# Token Blacklist (Redis with In-Memory Fallback)
# ---------------------------------------------------------------------------

async def add_token_to_blacklist(token: str, ttl: int | None = None) -> None:
    """Blacklist a JWT token in Redis with automatic TTL matching the token's expiry.
    Falls back to an in-memory set if Redis is disabled or unavailable.
    """
    calculated_ttl = ttl
    if calculated_ttl is None:
        try:
            # Extract exp claim without verifying signature to set precise TTL
            unverified = jwt.decode(token, options={"verify_signature": False})
            exp = unverified.get("exp")
            if exp:
                now_ts = datetime.now(timezone.utc).timestamp()
                remaining = int(exp - now_ts)
                calculated_ttl = max(remaining, 60) if remaining > 0 else 60
        except Exception:
            calculated_ttl = _DEFAULT_BLACKLIST_TTL_SECONDS

    if calculated_ttl is None:
        calculated_ttl = _DEFAULT_BLACKLIST_TTL_SECONDS

    # Always keep in memory fallback
    _MEMORY_BLACKLIST.add(token)

    redis_client = get_redis()
    if redis_client is not None:
        try:
            await redis_client.setex(f"{_BLACKLIST_PREFIX}{token}", calculated_ttl, "1")
            APP_LOGGER.info("Token successfully blacklisted in Redis with TTL=%ds", calculated_ttl)
        except Exception as e:
            APP_LOGGER.warning(
                "Redis blacklist write failed; falling back to in-memory store: %s", e
            )


async def is_token_blacklisted(token: str) -> bool:
    """Return True if the token has been revoked in Redis or the in-memory fallback."""
    if token in _MEMORY_BLACKLIST:
        return True

    redis_client = get_redis()
    if redis_client is not None:
        try:
            exists = await redis_client.exists(f"{_BLACKLIST_PREFIX}{token}")
            return exists == 1
        except Exception as e:
            APP_LOGGER.warning(
                "Redis blacklist check failed; checking in-memory fallback: %s", e
            )
            return token in _MEMORY_BLACKLIST

    return False


# ---------------------------------------------------------------------------
# Rate Limiting (Redis INCR + TTL with In-Memory Fallback)
# ---------------------------------------------------------------------------

async def is_rate_limited(
    ip: str,
    max_attempts: int = _DEFAULT_RATE_MAX_ATTEMPTS,
    window_seconds: int = _DEFAULT_RATE_WINDOW_SECONDS,
) -> bool:
    """Sliding-window rate limiter per IP.
    Uses Redis pipeline (INCR + TTL) when available; falls back to an in-memory
    timestamp list if Redis is unreachable.
    Returns True if the limit is exceeded.
    """
    redis_client = get_redis()
    if redis_client is not None:
        key = f"{_RATE_LIMIT_PREFIX}{ip}"
        try:
            async with redis_client.pipeline(transaction=True) as pipe:
                pipe.incr(key)
                pipe.ttl(key)
                results = await pipe.execute()

            count, current_ttl = results
            # If key has no expiration set (-1), initialize window TTL
            if current_ttl == -1:
                await redis_client.expire(key, window_seconds)

            return count > max_attempts
        except Exception as e:
            APP_LOGGER.warning(
                "Redis rate-limit check failed; using in-memory tracker: %s", e
            )

    # In-memory sliding-window fallback
    now = datetime.now(timezone.utc).timestamp()
    attempts = _MEMORY_RATE_LIMIT.get(ip, [])
    attempts = [ts for ts in attempts if now - ts < window_seconds]
    attempts.append(now)
    _MEMORY_RATE_LIMIT[ip] = attempts
    return len(attempts) > max_attempts


# ---------------------------------------------------------------------------
# JWT Helpers
# ---------------------------------------------------------------------------

def create_access_token(
    data: dict[str, Any],
    expires_delta: timedelta | None = None,
) -> str:
    """Create a signed access JWT."""
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire, "type": "access"})
    return jwt.encode(to_encode, settings.app.SECRET_KEY, algorithm="HS256")


def create_refresh_token(
    data: dict[str, Any],
    expires_delta: timedelta | None = None,
) -> str:
    """Create a signed refresh JWT (default 30-day expiry)."""
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(days=30))
    to_encode.update({"exp": expire, "type": "refresh"})
    return jwt.encode(to_encode, settings.app.SECRET_KEY, algorithm="HS256")


async def decode_jwt(token: str) -> dict[str, Any]:
    """Verify token signature, expiration, and Redis blacklist status.
    Raises UnauthenticatedError on any validation failure.
    """
    if await is_token_blacklisted(token):
        raise UnauthenticatedError(message="Token has been revoked. Please log in again.")

    try:
        payload = jwt.decode(token, settings.app.SECRET_KEY, algorithms=["HS256"])
        return payload
    except jwt.ExpiredSignatureError as e:
        raise UnauthenticatedError(message="Token has expired. Please log in again.") from e
    except jwt.PyJWTError as e:
        raise UnauthenticatedError(message=f"Invalid authentication token: {e}") from e


# ---------------------------------------------------------------------------
# FastAPI Dependency for Protected Endpoints
# ---------------------------------------------------------------------------

_bearer_scheme = HTTPBearer(auto_error=True)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer_scheme),
) -> dict[str, Any]:
    """FastAPI Dependency: extracts bearer token, validates against Redis blacklist
    and JWT signature/expiry, and returns the decoded payload.
    """
    token = credentials.credentials
    payload = await decode_jwt(token)
    if payload.get("type") != "access":
        raise UnauthenticatedError(message="Invalid token type: expected access token")
    return payload
