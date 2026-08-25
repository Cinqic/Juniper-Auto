"""Phase 1 training-support utilities: RNG/checkpoint state, checkpoint
save/load/validate, a tiny-batch overfit harness, and FLOWBOX hardware
profiling. Not a production training loop -- Phase 1 proves the
architecture and the training plumbing execute correctly, not a tuned
optimization recipe (see docs/architecture/precision-policy.md and
docs/phases/phase-1-architecture.md).
"""

from __future__ import annotations
