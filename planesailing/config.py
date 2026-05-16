"""Pipeline configuration: the tunables that shape acquisition and decoding.

Every PipelineConfig instance produces a stable fingerprint hash, which is
what links rows in `cosmos.readings` back to the exact configuration that
produced them. Changing any field bumps the fingerprint automatically — no
manual version tag to forget to update.
"""

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any


@dataclass(frozen=True)
class PipelineConfig:
    sample_rate: float = 2.4e6
    target_sr: float = 12e6
    n_reads_per_acquisition: int = 64
    n_reads_to_try_to_parse: int = 64
    n_samples_per_read: int = 4096
    sleep_length_s: float = 0.01
    sdr_gain: str = "auto"
    n: int = 1

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def fingerprint(self) -> str:
        canonical = json.dumps(self.as_dict(), sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
