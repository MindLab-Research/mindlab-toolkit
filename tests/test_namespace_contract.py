from __future__ import annotations

import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import mint  # noqa: E402


def test_namespaces_exist_with_explicit_exports() -> None:
    assert hasattr(mint, "tinker")
    assert hasattr(mint, "mint")
    assert isinstance(mint.__all__, list)
    assert isinstance(mint.tinker.__all__, list)
    assert isinstance(mint.mint.__all__, list)


def test_reexport_consistency_with_tinker_namespace() -> None:
    assert mint.ServiceClient is mint.tinker.ServiceClient
    assert mint.TrainingClient is mint.tinker.TrainingClient
    assert mint.SamplingClient is mint.tinker.SamplingClient
    assert mint.types is mint.tinker.types


def test_mint_namespace_does_not_redeclare_tinker_symbols() -> None:
    for symbol in ("ServiceClient", "TrainingClient", "SamplingClient"):
        assert symbol not in mint.mint.__all__


def test_tinker_version_is_locked() -> None:
    assert mint.__tinker_version__ == "0.6.0"


def test_mint_version_is_own_not_tinker() -> None:
    import tinker

    assert mint.__version__ != tinker.__version__
    assert mint.__version__ == mint.mint.MINT_VERSION


def test_version_guard_raises_on_mismatch() -> None:
    from mint.mint import EXPECTED_TINKER_VERSION, assert_tinker_version

    import tinker

    result = assert_tinker_version()
    assert result == EXPECTED_TINKER_VERSION
    assert result == tinker.__version__


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
