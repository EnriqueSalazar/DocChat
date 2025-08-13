"""DocChat - A CLI application for chatting with local documents.

Runtime bootstrap tweaks:
 - Force CPU-only execution (hide GPUs) to avoid PyTorch capability warnings.
 - Suppress specific noisy UserWarning about unsupported CUDA capabilities.
"""

from __future__ import annotations

import os
import warnings

# Hide all CUDA devices so torch never probes them (prevents capability warning emission)
if "CUDA_VISIBLE_DEVICES" not in os.environ:
	os.environ["CUDA_VISIBLE_DEVICES"] = ""  # no GPUs exposed

# Suppress the specific PyTorch warning if emitted later (defensive)
warnings.filterwarnings(
	"ignore",
	message=r".*not compatible with the current PyTorch installation.*",
	category=UserWarning,
)

# (Optional) eagerly import torch to ensure no warning slips through later imports
try:  # pragma: no cover - best effort
	import torch  # noqa: F401
except Exception:  # If torch not installed yet, ignore
	pass
