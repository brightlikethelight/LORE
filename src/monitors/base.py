"""Base classes and common utilities for monitors.

This module provides standardized error handling and result types
for all monitor implementations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from src.constants import DEFAULT_SUSPICION_THRESHOLD


class MonitorErrorCode(Enum):
    """Error codes for monitor failures."""

    NONE = "none"
    API_ERROR = "api_error"
    PARSE_ERROR = "parse_error"
    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit"
    VALIDATION_ERROR = "validation_error"
    UNKNOWN = "unknown"


@dataclass
class MonitorError:
    """Represents an error that occurred during monitoring.

    Attributes:
        code: Error code categorizing the failure type
        message: Human-readable error message
        details: Additional error details (e.g., exception info)
        recoverable: Whether the error is potentially recoverable with retry
    """

    code: MonitorErrorCode
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    recoverable: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "code": self.code.value,
            "message": self.message,
            "details": self.details,
            "recoverable": self.recoverable,
        }

    @classmethod
    def from_exception(cls, exc: Exception, recoverable: bool = True) -> MonitorError:
        """Create MonitorError from an exception.

        Args:
            exc: The exception that occurred
            recoverable: Whether the error is recoverable

        Returns:
            MonitorError instance
        """
        exc_type = type(exc).__name__

        # Map common exception types to error codes
        if "timeout" in exc_type.lower() or "timeout" in str(exc).lower():
            code = MonitorErrorCode.TIMEOUT
        elif "rate" in exc_type.lower() or "rate" in str(exc).lower():
            code = MonitorErrorCode.RATE_LIMIT
        elif "json" in exc_type.lower() or "parse" in exc_type.lower():
            code = MonitorErrorCode.PARSE_ERROR
        elif "api" in exc_type.lower() or "http" in exc_type.lower():
            code = MonitorErrorCode.API_ERROR
        else:
            code = MonitorErrorCode.UNKNOWN

        return cls(
            code=code,
            message=str(exc),
            details={"exception_type": exc_type},
            recoverable=recoverable,
        )


@dataclass
class MonitorResult:
    """Base result type for all monitors.

    This provides a consistent interface for handling both successful
    evaluations and failures across different monitor implementations.

    Attributes:
        success: Whether the evaluation completed successfully
        suspicion_score: Suspicion score (0-1), defaults to neutral on failure
        error: Error information if evaluation failed
        metadata: Additional result metadata
    """

    success: bool
    suspicion_score: float = DEFAULT_SUSPICION_THRESHOLD
    error: MonitorError | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and set defaults."""
        # Clamp suspicion score to valid range
        self.suspicion_score = max(0.0, min(1.0, self.suspicion_score))

        # Calculate stealth score
        self.stealth_score = 1.0 - self.suspicion_score

    @property
    def stealth_score(self) -> float:
        """Stealth score for Regime B fitness (1 - suspicion)."""
        return 1.0 - self.suspicion_score

    @stealth_score.setter
    def stealth_score(self, value: float) -> None:
        """Set stealth score (updates suspicion accordingly)."""
        # This is computed, so we don't actually store it
        pass

    @property
    def is_error(self) -> bool:
        """Check if this result represents an error."""
        return not self.success or self.error is not None

    @property
    def error_message(self) -> str | None:
        """Get error message if present."""
        return self.error.message if self.error else None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "success": self.success,
            "suspicion_score": self.suspicion_score,
            "stealth_score": self.stealth_score,
            "error": self.error.to_dict() if self.error else None,
            "metadata": self.metadata,
        }

    @classmethod
    def success_result(
        cls,
        suspicion_score: float,
        metadata: dict[str, Any] | None = None,
    ) -> MonitorResult:
        """Create a successful result.

        Args:
            suspicion_score: The computed suspicion score
            metadata: Optional additional metadata

        Returns:
            Successful MonitorResult
        """
        return cls(
            success=True,
            suspicion_score=suspicion_score,
            error=None,
            metadata=metadata or {},
        )

    @classmethod
    def failure_result(
        cls,
        error: MonitorError | Exception | str,
        neutral_score: float = DEFAULT_SUSPICION_THRESHOLD,
        metadata: dict[str, Any] | None = None,
    ) -> MonitorResult:
        """Create a failure result with neutral suspicion score.

        Args:
            error: Error information (MonitorError, Exception, or message)
            neutral_score: Suspicion score to use on failure (default: 0.5)
            metadata: Optional additional metadata

        Returns:
            Failed MonitorResult with neutral suspicion
        """
        if isinstance(error, MonitorError):
            monitor_error = error
        elif isinstance(error, Exception):
            monitor_error = MonitorError.from_exception(error)
        else:
            monitor_error = MonitorError(
                code=MonitorErrorCode.UNKNOWN,
                message=str(error),
            )

        return cls(
            success=False,
            suspicion_score=neutral_score,
            error=monitor_error,
            metadata=metadata or {},
        )


def handle_monitor_error(
    exc: Exception,
    default_score: float = DEFAULT_SUSPICION_THRESHOLD,
    context: str = "",
) -> MonitorResult:
    """Standard error handler for monitor implementations.

    This provides consistent error handling across all monitors,
    returning a neutral suspicion score on failure.

    Args:
        exc: The exception that occurred
        default_score: Default suspicion score on failure
        context: Additional context about where error occurred

    Returns:
        MonitorResult with error information and neutral score
    """
    metadata = {}
    if context:
        metadata["context"] = context

    return MonitorResult.failure_result(
        error=exc,
        neutral_score=default_score,
        metadata=metadata,
    )
