"""Validated upload-size handling from Northstar Cloud PR #284."""

from dataclasses import dataclass


MIB = 1024 * 1024
DEFAULT_UPLOAD_LIMIT_BYTES = 200 * MIB


@dataclass(frozen=True)
class UploadDecision:
    accepted: bool
    status_code: int
    message: str


def validate_upload_size(
    content_length: int | None,
    configured_limit_bytes: int = DEFAULT_UPLOAD_LIMIT_BYTES,
) -> UploadDecision:
    """Return a safe, predictable decision before reading an upload body."""
    if configured_limit_bytes <= 0:
        raise ValueError("configured_limit_bytes must be positive")
    if content_length is None:
        return UploadDecision(False, 411, "Content-Length is required")
    if content_length < 0:
        return UploadDecision(False, 400, "Content-Length cannot be negative")
    if content_length > configured_limit_bytes:
        limit_mib = configured_limit_bytes // MIB
        return UploadDecision(False, 413, f"File exceeds the {limit_mib} MiB upload limit")
    return UploadDecision(True, 200, "Upload size accepted")
