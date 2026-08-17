"""Pipeline configuration shared across stages."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import List, Optional


@dataclasses.dataclass
class PipelineConfig:
    """Configuration for a single source->target mapper fit.

    Defaults follow the paper's production recipe (Appendix E):
    500 FineWeb-Edu sequences x 1024 tokens, stride-4 subsampling,
    ridge lambda = 0.01.
    """

    source_model: str
    target_model: str
    workdir: str

    # Calibration corpus
    dataset: str = "HuggingFaceFW/fineweb-edu"
    dataset_config: Optional[str] = "sample-10BT"
    dataset_split: str = "train"
    text_file: Optional[str] = None  # local newline-delimited text overrides dataset
    num_sequences: int = 500
    seq_len: int = 1024
    subsample_stride: int = 4

    # Analysis / selection
    probe_tokens: int = 32768  # token subsample for the R^2 probe (cheap stage)
    top_k: int = 8             # source layers per target layer; "all" -> num source layers

    # Ridge fit
    ridge_lambda: float = 0.01

    # Diagnostics
    diag_sequences: int = 16   # held-out sequences for attention-output cosine
    diag_seq_len: int = 1024

    # Runtime
    device: str = "auto"       # auto | cuda | mps | cpu
    dtype: str = "auto"        # auto | bfloat16 | float16 | float32
    batch_size: int = 4
    seed: int = 0
    trust_remote_code: bool = False

    # ------------------------------------------------------------------
    def resolved_workdir(self) -> Path:
        p = Path(self.workdir).expanduser()
        p.mkdir(parents=True, exist_ok=True)
        return p

    def save(self) -> Path:
        path = self.resolved_workdir() / "config.json"
        path.write_text(json.dumps(dataclasses.asdict(self), indent=2))
        return path

    @classmethod
    def load(cls, workdir: str) -> "PipelineConfig":
        path = Path(workdir).expanduser() / "config.json"
        data = json.loads(path.read_text())
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


# Stage artifact locations, relative to the workdir --------------------------

def capture_dir(workdir: Path, role: str) -> Path:
    """role is 'source' or 'target'."""
    return workdir / f"capture_{role}"


def analysis_path(workdir: Path) -> Path:
    return workdir / "analysis.json"


def heatmap_path(workdir: Path) -> Path:
    return workdir / "r2_heatmaps.png"


def mapper_path(workdir: Path) -> Path:
    return workdir / "mapper.safetensors"


def diagnostics_path(workdir: Path) -> Path:
    return workdir / "diagnostics.json"
