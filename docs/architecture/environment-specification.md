# Phase 0 Environment Specification

This records the actual environment Phase 0 was engineered and validated
in, captured directly from the machine rather than guessed. It is not a
requirement that every future contributor use this exact machine -- FLOWBOX
is the baseline consumer-hardware target (see
[project-charter](../research/project-charter.md)), and CI validates on a
different (cloud, CPU-only) Linux environment on purpose, to prove Phase 0
does not secretly depend on FLOWBOX specifics.

## FLOWBOX (development baseline)

Captured 2026-08-25.

| Field | Value |
|---|---|
| Hostname | FLOWBOX |
| Distribution | Linux Mint 22.3 (Zena), `ID_LIKE=ubuntu debian` |
| Kernel | `7.0.0-30-generic` (Ubuntu 24.04-based, x86_64) |
| CPU | AMD Ryzen 7 5700G with Radeon Graphics (8 cores / 16 threads) |
| RAM | 16 GB installed (~14 GiB usable per `free -h`) |
| GPU | NVIDIA GeForce RTX 2060, 6144 MiB VRAM |
| NVIDIA driver | 595.84 |
| CUDA toolkit (`nvcc`) | not installed on the host -- PyTorch ships its own bundled CUDA runtime libraries as pip dependencies (see below), so a system-wide CUDA toolkit install is not required for this project |
| Primary storage | NVMe, ~234 GB partition (`/`), ~175 GB free at capture time |
| Secondary storage | not present in this capture (FLOWBOX's documented 500 GB HDD was not mounted at capture time; core Phase 0 work does not require it) |
| Git | 2.43.0 |
| GitHub CLI (`gh`) | authenticated, account `Cinqic`, scopes `gist, read:org, repo, workflow` |

## Canonical supported Python version

**Python 3.12** (`requires-python = ">=3.12,<3.13"` in `pyproject.toml`).
FLOWBOX ships Python 3.12.3 as `python3`; no separate interpreter install was
required.

`pip` and `venv` are available via the standard library
(`python3 -m venv`, `python3 -m ensurepip`); FLOWBOX did not have a
standalone global `pip3` binary, which is why the recovery procedure
(`docs/recovery/`) bootstraps pip inside a fresh virtual environment rather
than assuming a global `pip3`.

## Dependency manager

`pip` + `pip-tools` (`pip-compile`), inside a `venv`. See
[ADR-0004](../adr/0004-dependency-locking-approach.md) for why this was
chosen over `uv` (not present on FLOWBOX at Phase 0 engineering time) or
Poetry.

## Key pinned versions (from `requirements-lock.txt`)

| Package | Version |
|---|---|
| torch | 2.13.0 (`+cu130` build on FLOWBOX; CPU execution is exercised explicitly in tests and does not require this build tag) |
| pydantic | 2.13.4 |
| PyYAML | 6.0.3 |
| numpy | 2.5.2 |
| pytest | 8.4.2 |
| pip-tools | 7.6.1 |

Exact hashes for every direct and transitive dependency are in
`requirements-lock.txt` at the repository root.

## What CI validates instead

GitHub Actions CI (`.github/workflows/`) runs on `ubuntu-latest`
(GitHub-hosted, CPU-only, not FLOWBOX), installing from the same
`requirements-lock.txt`. This is deliberate: if Phase 0 only ever validated
on FLOWBOX, it would not actually prove the project survives loss of the
current local machine. See
[ADR-0006](../adr/0006-validation-and-recovery-strategy.md).

## Known limitations of this capture

- The secondary 500 GB HDD described in the project charter was not
  mounted/verified at capture time; nothing in Phase 0 depends on it.
- `nvcc`/full CUDA toolkit is not installed at the OS level; this is
  accepted for Phase 0 since PyTorch's pip wheel bundles the CUDA runtime
  libraries it needs. A future phase doing custom CUDA kernel work would
  need to revisit this.
- This is a single point-in-time capture (2026-08-25), not a continuously
  updated environment monitor.
