from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ErrorPayload:
    code: str
    message: str
    status_code: int = 400


class WaggleError(Exception):
    def __init__(self, code: str, message: str, *, status_code: int = 400) -> None:
        super().__init__(message)
        self.payload = ErrorPayload(code=code, message=message, status_code=status_code)

    @property
    def code(self) -> str:
        return self.payload.code

    @property
    def status_code(self) -> int:
        return self.payload.status_code


class AuthenticationError(WaggleError):
    def __init__(self, message: str = "Authentication failed.") -> None:
        super().__init__("authentication_failed", message, status_code=401)


class AuthorizationError(WaggleError):
    def __init__(self, message: str = "Not authorized to perform this action.") -> None:
        super().__init__("authorization_failed", message, status_code=403)


class RateLimitExceededError(WaggleError):
    def __init__(self, message: str = "Rate limit exceeded.") -> None:
        super().__init__("rate_limited", message, status_code=429)


class PayloadTooLargeError(WaggleError):
    def __init__(self, message: str = "Payload too large.") -> None:
        super().__init__("payload_too_large", message, status_code=413)


class ServiceUnavailableError(WaggleError):
    def __init__(self, message: str = "Service unavailable.") -> None:
        super().__init__("service_unavailable", message, status_code=503)


class ValidationFailure(WaggleError):
    def __init__(self, message: str) -> None:
        super().__init__("validation_failed", message, status_code=400)


class ConflictResolutionError(WaggleError):
    def __init__(self, message: str) -> None:
        super().__init__("conflict_resolution_failed", message, status_code=400)


class DanglingEdgeError(WaggleError):
    def __init__(self, message: str) -> None:
        super().__init__("dangling_edge", message, status_code=400)


class HashVerificationError(WaggleError):
    def __init__(self, message: str) -> None:
        super().__init__("hash_verification_failed", message, status_code=400)


class SchemaVersionError(WaggleError):
    def __init__(self, message: str) -> None:
        super().__init__("schema_version_incompatible", message, status_code=400)
