"""Fail-fast validation for the installed Tinker distribution."""

from __future__ import annotations

import re
from importlib import metadata

EXPECTED_TINKER_VERSION = "0.22.0"
_VERSION_PATTERN = re.compile(r"^\d+(?:\.\d+)+(?:[A-Za-z0-9.+_-]*)?$")


def _remediation(detail: str) -> RuntimeError:
    return RuntimeError(
        f"mindlab-toolkit requires tinker=={EXPECTED_TINKER_VERSION}, but {detail}\n"
        f"Please install tinker=={EXPECTED_TINKER_VERSION}.\n"
        "Do not use pip install --no-deps."
    )


def check_tinker_version() -> str:
    """Return the installed supported version or raise an actionable error."""
    try:
        installed = metadata.version("tinker")
    except metadata.PackageNotFoundError as exc:
        raise _remediation("tinker is not installed.") from exc
    except Exception as exc:
        raise _remediation("the installed tinker version could not be read.") from exc

    if not isinstance(installed, str) or not _VERSION_PATTERN.fullmatch(installed):
        raise _remediation("could not read a valid installed tinker version.")
    if installed != EXPECTED_TINKER_VERSION:
        raise _remediation(f"tinker=={installed} is installed.")
    return installed


__all__ = ["EXPECTED_TINKER_VERSION", "check_tinker_version"]
