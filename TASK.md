# MinT 0.2.0 / Tinker 0.22.0 Compatibility Tasks

## Goal

Deliver a test-driven compatibility release with four user-facing guarantees:

1. A normal MinT installation installs `tinker==0.22.0` automatically.
2. `import mint` fails immediately with an actionable error when the installed
   Tinker version is missing or incompatible.
3. Every standard public client exposed by Tinker 0.22.0 supports MinT `sk-`
   keys after MinT is imported.
4. Importing Tinker without MinT preserves Tinker's original key behavior.

The release must also use the current MinT training endpoints:

- Global: `https://mint.macaron.im/train`
- China: `https://mintcn.macaron.xin/train`

An explicit `MINT_BASE_URL` must override either regional default.

## User Journey

### Normal installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install mindlab-toolkit==0.2.0
```

Expected installed distributions:

```text
mindlab-toolkit==0.2.0
tinker==0.22.0
```

The user then configures MinT:

```bash
export MINT_API_KEY=sk-...
export MINT_BASE_URL=https://mint.macaron.im/train
```

China deployment:

```bash
export MINT_API_KEY=sk-...
export MINT_BASE_URL=https://mintcn.macaron.xin/train
```

Application code:

```python
import mint

client = mint.ServiceClient()
```

Expected behavior:

- `import mint` verifies the installed Tinker distribution version.
- `MINT_API_KEY` is propagated to Tinker authentication.
- `sk-` is not rejected by Tinker's local `tml-` prefix validation.
- Requests carry `X-API-Key: sk-...`.
- Authorization is ultimately decided by the MinT server.

### Invalid installation

For an installation made with `pip install --no-deps` or an environment that
contains an incompatible Tinker version, `import mint` must fail before any
Tinker-internal MinT module is imported. The error should be equivalent to:

```text
RuntimeError: mindlab-toolkit requires tinker==0.22.0, but tinker==0.21.0 is installed.
Please install tinker==0.22.0.
Do not use pip install --no-deps.
```

### Import boundary

```python
import tinker
```

Preserves original Tinker behavior:

- `tml-` is accepted.
- `eyJ...` is accepted.
- `sk-` is rejected.

```python
import mint as tinker
```

or:

```python
import mint
import tinker
```

Loads the MinT compatibility patch, so standard Tinker 0.22.0 client paths can
use `sk-` keys. The original
`tinker.lib._auth_token_provider.ApiKeyAuthProvider` class must still reject
`sk-` when called directly.

## Execution Rules

- Follow red-green-refactor: add a failing acceptance test before each behavior
  change, implement the minimum fix, then rerun the focused and full suites.
- Test user-visible behavior first. Keep implementation-binding assertions to
  the minimum needed to protect the compatibility boundary.
- Scope support to Tinker 0.22.0. Do not claim compatibility with future Tinker
  versions.
- Do not commit OpenSpec files, reports, temporary scripts, virtual
  environments, generated build directories, or local patch files.
- Before every commit, inspect `git diff --cached --name-only`.

## Phase 1: Establish the Baseline

- [ ] Create a new branch from the latest `origin/main`.
  - Verify: `git log -1` is the expected post-`v2.6.5` baseline.
- [ ] Record the Python package version and repository tag version separately.
  - Verify: document that package `0.2.0` and repository tag `v2.6.5` are
    different version systems unless the release process is changed.
- [ ] Confirm `pyproject.toml` declares exactly `tinker==0.22.0`.
  - Verify: no other dependency file declares another Tinker version.
- [ ] Scan the Tinker 0.22.0 production source for all
  `ApiKeyAuthProvider` constructions and `resolve_auth_provider` calls.
  - Verify: classify `_auth_token_provider`, `InternalClientHolder`, and
    `AsyncTinker` paths; exclude tests and documentation from production paths.
- [ ] Inventory all standard public client classes exported by Tinker 0.22.0.
  - Verify: produce the exact client matrix used by Phase 5 tests.

## Phase 2: Runtime Version Contract

### Red tests

- [ ] Add a subprocess acceptance test where installed Tinker is `0.22.0`.
  - Expect: `import mint` succeeds.
- [ ] Add a subprocess acceptance test that reports installed Tinker as
  `0.21.0`.
  - Expect: `import mint` raises the documented `RuntimeError`.
- [ ] Add a test for missing Tinker distribution metadata.
  - Expect: an actionable `RuntimeError`, not a partial-import
    `ModuleNotFoundError`.
- [ ] Add a test for unreadable or malformed version metadata.
  - Expect: an actionable `RuntimeError` naming the expected version.
- [ ] Add an import-order test.
  - Expect: version validation runs before importing `_auth.py`, `_mintx.py`,
    or any other MinT module that imports Tinker internals.

### Implementation

- [ ] Add a lightweight version-check module that imports no Tinker internals.
- [ ] Define `EXPECTED_TINKER_VERSION = "0.22.0"` in that lightweight module.
- [ ] Read the installed version with
  `importlib.metadata.version("tinker")`.
- [ ] Convert `PackageNotFoundError` and invalid metadata into the documented
  `RuntimeError`.
- [ ] Compare the installed version strictly with `0.22.0`.
- [ ] Call the version check at the beginning of `src/mint/__init__.py`.
  - Verify: it runs before importing `mint.mint`, `_auth`, `_mintx`, or the
    Tinker compatibility exports.
- [ ] Refactor only enough to keep the initialization order explicit.

## Phase 3: Installation Contract

### Red tests

- [ ] Build a wheel and inspect its metadata.
  - Expect: `Requires-Dist: tinker==0.22.0`.
- [ ] In a new venv, run a normal installation from the built wheel or source.
  - Expect: `importlib.metadata.version("tinker") == "0.22.0"` and
    `import mint` succeeds.
- [ ] In another new venv, install a different Tinker version first and then
  perform a normal MinT installation.
  - Expect: pip resolves/replaces Tinker with `0.22.0`.
- [ ] In another new venv, install MinT with `--no-deps` and no Tinker.
  - Expect: `import mint` gives the explicit dependency/version error.
- [ ] In another new venv, install MinT with `--no-deps` and an incompatible
  Tinker version.
  - Expect: `import mint` gives the explicit version error.

### Environment requirements

- [ ] Use a package source that contains every Tinker 0.22.0 dependency,
  including `pyqwest>=0.4.1`, for the normal-install acceptance test.
- [ ] If the configured mirror lacks `pyqwest`, report that mirror failure as
  an environment blocker; do not substitute `--no-deps` and claim that the
  normal installation contract passed.
- [ ] Record the exact index URL or internal wheel source used by CI without
  committing credentials.

## Phase 4: Base URL Contract

### Product decisions to lock before implementation

- [ ] Confirm the automatic regional selection mechanism.
  - Proposed default: global endpoint
    `https://mint.macaron.im/train`.
  - Proposed China selection: explicit `MINT_BASE_URL` set to
    `https://mintcn.macaron.xin/train`.
  - If automatic region configuration is required, define a stable variable
    such as `MINT_REGION=global|cn` before writing implementation tests.
- [ ] Confirm trailing-slash behavior.
  - Expect: client URL joining does not duplicate or remove the `/train`
    deployment prefix.

### Red tests

- [ ] With no Base URL variables, test the agreed default endpoint.
- [ ] With `MINT_BASE_URL=https://mint.macaron.im/train`, expect
  `TINKER_BASE_URL` to receive the global endpoint.
- [ ] With `MINT_BASE_URL=https://mintcn.macaron.xin/train`, expect
  `TINKER_BASE_URL` to receive the China endpoint.
- [ ] With both `MINT_BASE_URL` and `TINKER_BASE_URL`, expect
  `MINT_BASE_URL` to win for the MinT process.
- [ ] With only an explicit constructor `base_url`, expect the constructor
  argument to win over environment defaults.
- [ ] Test repeated imports/patch application.
  - Expect: URL synchronization is deterministic and idempotent.
- [ ] Add request-path tests using a mock transport.
  - Expect: requests retain the `/train` prefix for both endpoints.

### Implementation

- [ ] Replace the obsolete `https://mint.macaron.xin` default.
- [ ] Define named constants for the global and China training endpoints.
- [ ] Implement only the regional selection behavior approved above.
- [ ] Preserve explicit constructor and `MINT_BASE_URL` precedence.

## Phase 5: Authentication Behavior Contract

### Provider behavior

- [ ] Test `sk-test` through `resolve_auth_provider` with
  `enforce_cmd=False` and `True`.
  - Expect: `MintApiKeyAuthProvider`.
- [ ] Test `tml-test` through both enforcement modes.
  - Expect: original Tinker behavior, including credential-command semantics.
- [ ] Test `eyJ-test` through both enforcement modes.
  - Expect: original Tinker provider behavior.
- [ ] Test `abc-test` through both enforcement modes.
  - Expect: original Tinker validation error.
- [ ] Test the `InternalClientHolder` direct provider path.
- [ ] Test the `AsyncTinker` direct provider path.
- [ ] Test all other production call sites found in the Phase 1 scan.
- [ ] Verify the original Tinker `ApiKeyAuthProvider(api_key="sk-test")`
  still raises Tinker's prefix error.

### Header behavior

- [ ] Send a request through each auth provider with `httpx.MockTransport` or
  an equivalent mock.
- [ ] Verify `X-API-Key` equals the exact user-supplied token for `sk-`,
  `tml-`, and `eyJ` paths.
- [ ] Verify no real network or real API key is required by these tests.

### Environment behavior

- [ ] Test explicit `api_key` input.
- [ ] Test `TINKER_API_KEY` input.
- [ ] Test `MINT_API_KEY` synchronization.
- [ ] Define and test `MINT_API_KEY=""` precedence.
  - Proposed behavior: an empty MinT value is treated as unset and does not
    overwrite a valid `TINKER_API_KEY`.

## Phase 6: Boundary Contract

Lock each expectation before implementing it:

- [ ] `api_key=None` with a valid environment key falls back to that key.
- [ ] `api_key=None` without an environment key raises the original Tinker
  error.
- [ ] `api_key=""` follows Tinker's current fallback-to-environment behavior.
- [ ] `api_key="abc-test"` uses original Tinker validation and fails.
- [ ] `api_key="sk"` is not a MinT key and fails original validation.
- [ ] `api_key="sk-"` is accepted if prefix-only validation remains the
  product decision; otherwise define the minimum payload rule first.
- [ ] Case variants such as `Sk-test` and `TMl-test` receive no special
  handling.
- [ ] Non-string values do not enter the MinT provider path and retain the
  original Tinker/Python error behavior.
- [ ] Applying the patch repeatedly does not stack wrappers.
- [ ] Importing MinT repeatedly is idempotent.

## Phase 7: Public Client Contract

For every standard public client identified in Phase 1, add one construction
smoke test rather than duplicating every API method test.

- [ ] `ServiceClient` construction with explicit `sk-` key.
- [ ] `ServiceClient` construction with `MINT_API_KEY`.
- [ ] `TrainingClient` construction through its public creation path.
- [ ] `SamplingClient` construction through its public creation path.
- [ ] `AsyncTinker` construction with explicit and environment `sk-` keys.
- [ ] Any additional standard public client exported by Tinker 0.22.0.
- [ ] Stub server configuration and network calls with mock transports.
- [ ] Verify each client reaches the common compatible auth layer.
- [ ] Verify `tml-` and `eyJ` are not converted to a MinT provider.

## Phase 8: Import Contract

Run each scenario in an isolated subprocess:

- [ ] `import tinker` only, then direct original provider with `sk-`.
  - Expect: rejected.
- [ ] `import mint`, then create a MinT-reexported standard client with `sk-`.
  - Expect: accepted locally.
- [ ] `import mint as tinker`, then create `tinker.ServiceClient()` with `sk-`.
  - Expect: accepted locally.
- [ ] `import tinker` followed by `import mint`, then use the standard client
  path with `sk-`.
  - Expect: the MinT compatibility bindings are applied to the already-loaded
    Tinker modules.
- [ ] Recheck that the original provider class still rejects `sk-` after MinT
  import.

## Phase 9: README Contract

- [ ] Replace every stale Tinker 0.15.0 reference with 0.22.0 where applicable.
- [ ] Document that normal pip installation pins and installs Tinker 0.22.0.
- [ ] Document that `pip install --no-deps` bypasses dependency installation.
- [ ] Document the runtime version check and its exact remediation.
- [ ] Document the actual package installation source:
  - use `pip install mindlab-toolkit==0.2.0` only if it is published on the
    user's configured package index;
  - otherwise document source or Git installation accurately.
- [ ] Document both training endpoints:
  - global: `https://mint.macaron.im/train`;
  - China: `https://mintcn.macaron.xin/train`.
- [ ] Document `MINT_BASE_URL` precedence and the approved regional selection
  mechanism.
- [ ] Document `MINT_API_KEY` setup.
- [ ] Document `import mint`, `import mint as tinker`, and `import tinker`
  behavior separately.
- [ ] Explain that direct use of the original Tinker auth provider still
  rejects `sk-` by design.
- [ ] Explain the difference between package version `0.2.0` and repository
  release tags, or align them before release.

## Phase 10: Full Verification

- [ ] Run all focused runtime-version tests.
- [ ] Run all authentication and header tests.
- [ ] Run all public-client smoke tests.
- [ ] Run all isolated import tests.
- [ ] Run the full unit test suite.
- [ ] Run `python -m compileall -q src tests`.
- [ ] Run `python -m build`.
- [ ] Inspect wheel metadata for `tinker==0.22.0`.
- [ ] Run the normal-install test in a brand-new venv with full dependency
  resolution.
- [ ] Run the wrong-version replacement test in another brand-new venv.
- [ ] Run the `--no-deps` failure tests in separate brand-new venvs.
- [ ] Run mock requests against both Base URL configurations and verify both
  the `/train` prefix and `X-API-Key` header.
- [ ] Confirm no test relies on a real API key or production network request.

## Phase 11: Clean PR and Release

- [ ] Inspect `git status --short` before staging.
- [ ] Stage explicit files only; never use a broad add command in the existing
  dirty worktree.
- [ ] Confirm `git diff --cached --name-only` contains only task-related code,
  tests, README, and package metadata.
- [ ] Confirm no `openspec/`, report, temporary script, venv, build artifact,
  or local patch file is staged.
- [ ] Push a dedicated branch and create a focused PR.
- [ ] Verify the remote PR file list with `gh pr view <PR> --json files`.
- [ ] Require all installation, version, auth, header, URL, client, and import
  contracts to pass before merge.
- [ ] Decide whether the Python package version must move from `0.2.0` to
  `0.2.1`; do not create a release tag until this is explicit.
- [ ] Create the next release tag only from the verified merge commit.

## Definition of Done

- [ ] A normal installation resolves `tinker==0.22.0` automatically.
- [ ] Wheel metadata pins `tinker==0.22.0`.
- [ ] An incompatible or missing Tinker installation fails immediately during
  `import mint` with an actionable error.
- [ ] Version validation happens before importing MinT modules that depend on
  Tinker internals.
- [ ] The global training endpoint is available at
  `https://mint.macaron.im/train`.
- [ ] The China training endpoint is available at
  `https://mintcn.macaron.xin/train`.
- [ ] Explicit Base URL configuration has documented and tested precedence.
- [ ] All standard public Tinker 0.22.0 client paths support `sk-` after MinT
  import.
- [ ] `tml-` and `eyJ` retain original Tinker behavior.
- [ ] Original Tinker-only usage still rejects `sk-`.
- [ ] Requests carry the exact configured token in `X-API-Key`.
- [ ] The full clean-environment test matrix passes.
- [ ] README matches actual installation, Base URL, import, and authentication
  behavior.
- [ ] The PR contains only files directly required by this task.
