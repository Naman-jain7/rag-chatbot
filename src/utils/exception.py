from typing import Dict, Optional, Any
from src.utils.logger import APP_LOGGER


class AppException(Exception):
    """
    Base exception for the application.
    All custom exceptions must inherit from this.
    """

    def __init__(
        self,
        message: str,
        error_code: str = "INTERNAL_SERVER_ERROR",
        status_code: int = 500,
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        self.payload = payload or {}

        APP_LOGGER.error(
            f"[{self.error_code}] {self.message} | Payload: {self.payload}",
            exc_info=True,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the error for API responses."""
        return {
            "error": {
                "message": self.message,
                "code": self.error_code,
                "details": self.payload,
            }
        }


class ValidationError(AppException):
    """Raised when input validation fails."""

    def __init__(self, message: str, payload: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(
            message=message,
            error_code="VALIDATION_ERROR",
            status_code=400,
            payload=payload,
        )


class ResourceNotFoundError(AppException):
    """Raised when a requested resource does not exist."""

    def __init__(self, message: str, payload: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(
            message=message, error_code="NOT_FOUND", status_code=404, payload=payload
        )


class UnauthenticatedError(AppException):
    """Raised when authentication fails or is missing."""

    def __init__(
        self,
        message: str = "Authentication required",
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message=message,
            error_code="UNAUTHENTICATED",
            status_code=401,
            payload=payload,
        )


class PermissionDeniedError(AppException):
    """Raised when an authenticated user lacks permissions."""

    def __init__(
        self,
        message: str = "Permission denied",
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message=message,
            error_code="PERMISSION_DENIED",
            status_code=403,
            payload=payload,
        )


class ConflictError(AppException):
    """Raised when a resource conflict occurs (e.g., duplicate unique field)."""

    def __init__(self, message: str, payload: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(
            message=message, error_code="CONFLICT", status_code=409, payload=payload
        )


class ExternalServiceError(AppException):
    """Raised when a third-party dependency or external API fails."""

    def __init__(
        self,
        message: str = "External service unavailable",
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message=message,
            error_code="EXTERNAL_SERVICE_ERROR",
            status_code=502,
            payload=payload,
        )


class RateLimitExceededError(AppException):
    """Raised when the user has sent too many requests in a given amount of time."""

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message=message,
            error_code="TOO_MANY_REQUESTS",
            status_code=429,
            payload=payload,
        )


class AllProvidersExhaustedError(AppException):
    def __init__(
        self,
        message: str = "Rate limit exceeded",
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message=message,
            error_code="ALL_PROVIDERS_EXHAUSTED",
            status_code=420,
            payload=payload,
        )
