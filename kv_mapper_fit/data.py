"""Calibration corpus loading and tokenization.

Default corpus is FineWeb-Edu (streaming, no full download), matching the
paper. A local newline-delimited text file can be substituted with
--text-file, which also enables fully offline runs and tests.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, List

import torch

from kv_mapper_fit.config import PipelineConfig


def _iter_texts(cfg: PipelineConfig) -> Iterator[str]:
    if cfg.text_file:
        for line in Path(cfg.text_file).expanduser().read_text().splitlines():
            line = line.strip()
            if line:
                yield line
        return
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Install the 'data' extra (pip install 'kv-mapper-fit[data]') to "
            "stream FineWeb-Edu, or pass --text-file."
        ) from exc
    ds = load_dataset(
        cfg.dataset, cfg.dataset_config, split=cfg.dataset_split, streaming=True
    )
    for row in ds:
        yield row["text"]


def build_calibration_batches(
    cfg: PipelineConfig, tokenizer, num_sequences: int, seq_len: int,
    skip_sequences: int = 0,
) -> List[torch.Tensor]:
    """Tokenize the corpus into fixed-length sequences and group into batches.

    Documents shorter than seq_len are packed together (concatenated) so every
    calibration sequence is exactly seq_len tokens, matching the paper's
    fixed-length protocol. skip_sequences lets the diagnostics stage draw
    held-out sequences from beyond the calibration range.

    Returns a list of [batch, seq_len] int64 tensors.
    """
    sequences: List[List[int]] = []
    buffer: List[int] = []
    needed = num_sequences + skip_sequences
    for text in _iter_texts(cfg):
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        buffer.extend(ids)
        while len(buffer) >= seq_len:
            sequences.append(buffer[:seq_len])
            buffer = buffer[seq_len:]
            if len(sequences) >= needed:
                break
        if len(sequences) >= needed:
            break
    if len(sequences) < needed:
        raise RuntimeError(
            f"Corpus produced only {len(sequences)} sequences of length "
            f"{seq_len}; need {needed}. Provide a larger corpus or reduce "
            "--num-sequences / --seq-len."
        )
    sequences = sequences[skip_sequences:needed]

    batches = []
    for start in range(0, len(sequences), cfg.batch_size):
        chunk = sequences[start:start + cfg.batch_size]
        batches.append(torch.tensor(chunk, dtype=torch.long))
    return batches
