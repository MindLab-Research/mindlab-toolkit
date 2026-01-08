"""
Mind Lab Toolkit (MinT) - the Open Infrastructure for Experiential Intelligence.

All tinker APIs are available directly:
    import mint
    client = mint.TrainingClient(...)

MinT extends tinker with additional functionality while maintaining
full backward compatibility.
"""
import os as _os

# Configure mint defaults before importing tinker
# MINT_API_KEY takes precedence, falls back to TINKER_API_KEY
if "MINT_API_KEY" in _os.environ and "TINKER_API_KEY" not in _os.environ:
    _os.environ["TINKER_API_KEY"] = _os.environ["MINT_API_KEY"]

# MINT_BASE_URL takes precedence, falls back to TINKER_BASE_URL, then mint default
if "MINT_BASE_URL" in _os.environ:
    _os.environ["TINKER_BASE_URL"] = _os.environ["MINT_BASE_URL"]
elif "TINKER_BASE_URL" not in _os.environ:
    _os.environ["TINKER_BASE_URL"] = "https://mint.macaron.im"

_MINT_VERSION = "0.1.0"

# Inject Mint User-Agent header to identify as Mint client
import tinker.lib.public_interfaces.service_client as _service_client_module

_MINT_HEADERS = {"User-Agent": f"Mint/Python {_MINT_VERSION}"}
_original_get_headers = _service_client_module._get_default_headers


def _mint_get_default_headers():
    return {**_original_get_headers(), **_MINT_HEADERS}


_service_client_module._get_default_headers = _mint_get_default_headers

# Monkey-patch SDK to accept mint:// paths
from tinker.lib.internal_client_holder import InternalClientHolder
from tinker.lib.client_connection_pool_type import ClientConnectionPoolType
from tinker import types as _types

_original_create_sampling_session = InternalClientHolder._create_sampling_session


async def _mint_create_sampling_session(self, model_path=None, base_model=None):
    """Patched version: no validation, pass path directly to server."""
    sampling_session_seq_id = self._sampling_client_counter
    self._sampling_client_counter += 1
    with self.aclient(ClientConnectionPoolType.SESSION) as client:
        request = _types.CreateSamplingSessionRequest(
            session_id=self._session_id,
            sampling_session_seq_id=sampling_session_seq_id,
            model_path=model_path,
            base_model=base_model,
        )
        result = await client.service.create_sampling_session(request=request)
        return result.sampling_session_id


InternalClientHolder._create_sampling_session = _mint_create_sampling_session

# Re-export everything from tinker
from tinker import *
from tinker import __all__ as _tinker_all
from tinker import __version__ as _tinker_version

__tinker_version__ = _tinker_version
__version__ = _MINT_VERSION  # Restore after tinker import

__all__ = [
    *_tinker_all,
    "__version__",
    "__tinker_version__",
]
