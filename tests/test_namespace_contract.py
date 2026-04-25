from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import mint  # noqa: E402
from mint import mint as mint_impl  # noqa: E402


def test_namespaces_exist_with_explicit_exports() -> None:
    import tinker

    assert hasattr(mint, "tinker")
    assert hasattr(mint, "mint")
    assert isinstance(mint.__all__, list)
    assert isinstance(mint.tinker.__all__, list)
    assert isinstance(mint.mint.__all__, list)
    assert mint.tinker.__all__ == list(tinker.__all__)


def test_reexport_consistency_with_tinker_namespace() -> None:
    assert mint.ServiceClient is mint.tinker.ServiceClient
    assert mint.TrainingClient is mint.tinker.TrainingClient
    assert mint.SamplingClient is mint.tinker.SamplingClient
    assert mint.types is mint.tinker.types


def test_mint_namespace_does_not_redeclare_tinker_symbols() -> None:
    for symbol in ("ServiceClient", "TrainingClient", "SamplingClient"):
        assert symbol not in mint.mint.__all__


def test_tinker_version_tracks_installed_version() -> None:
    import tinker

    assert mint.__tinker_version__ == tinker.__version__


def test_mint_version_is_own_not_tinker() -> None:
    import tinker

    assert mint.__version__ != tinker.__version__
    assert mint.__version__ == mint.mint.MINT_VERSION


def test_capability_guard_accepts_current_install() -> None:
    from mint.mint import (
        EXPECTED_TINKER_VERSION,
        SUPPORTED_TINKER_SPEC,
        SUPPORTED_TINKER_VERSIONS,
        assert_tinker_compat,
        assert_tinker_version,
    )

    import tinker

    result = assert_tinker_compat()
    assert result == tinker.__version__
    assert assert_tinker_version() == tinker.__version__
    assert EXPECTED_TINKER_VERSION in SUPPORTED_TINKER_VERSIONS
    assert SUPPORTED_TINKER_SPEC == f"=={EXPECTED_TINKER_VERSION}"


def test_tinker_version_guard_rejects_unsupported_without_override(monkeypatch) -> None:
    from mint.mint import _assert_supported_tinker_version

    monkeypatch.delenv("MINT_ALLOW_UNSUPPORTED_TINKER", raising=False)
    fake_tinker = SimpleNamespace(__version__="999.0.0", __file__="/tmp/fake-tinker.py")

    try:
        _assert_supported_tinker_version(fake_tinker)
    except RuntimeError as exc:
        message = str(exc)
    else:
        raise AssertionError("unsupported Tinker version was accepted without override")

    assert "requires a validated Tinker SDK version" in message
    assert "Installed tinker version: 999.0.0" in message
    assert "python -m pip install --force-reinstall 'tinker==" in message
    assert "/tmp/fake-tinker.py" in message


def test_sync_env_sets_default_base_url_when_unset(monkeypatch) -> None:
    monkeypatch.delenv("MINT_BASE_URL", raising=False)
    monkeypatch.delenv("TINKER_BASE_URL", raising=False)
    monkeypatch.delenv("MINT_API_KEY", raising=False)

    mint_impl.sync_env()

    assert os.environ["TINKER_BASE_URL"] == "https://mint.macaron.xin"


def test_no_wildcard_imports_in_namespace_files() -> None:
    package_root = SRC_ROOT / "mint"
    files = [
        package_root / "__init__.py",
        package_root / "tinker" / "__init__.py",
        package_root / "mint" / "__init__.py",
    ]
    for path in files:
        text = path.read_text(encoding="utf-8")
        assert "import *" not in text, f"wildcard import found in {path}"
