"""Regression tests saved with Northstar Cloud PR #284."""

import pytest

from upload_handler import MIB, validate_upload_size


def test_accepts_file_below_limit() -> None:
    assert validate_upload_size(199 * MIB).accepted is True


def test_accepts_file_exactly_at_limit() -> None:
    decision = validate_upload_size(200 * MIB)
    assert decision.accepted is True
    assert decision.status_code == 200


def test_rejects_file_above_limit_with_clear_message() -> None:
    decision = validate_upload_size((200 * MIB) + 1)
    assert decision.accepted is False
    assert decision.status_code == 413
    assert "200 MiB" in decision.message


def test_rejects_missing_content_length() -> None:
    assert validate_upload_size(None).status_code == 411


def test_rejects_invalid_configured_limit() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        validate_upload_size(10, configured_limit_bytes=0)
