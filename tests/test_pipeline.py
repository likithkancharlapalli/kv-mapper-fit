"""End-to-end pipeline smoke tests on tiny random models (fully offline).

Two checks:
  1. Self-transfer (source == target): the identity relationship is exactly
     linear, so probe R^2, ridge R^2, eval R^2, and attention-output cosine
     must all be ~1. This validates the math end to end (RoPE strip/apply
     round trip, centering, ridge solve, export/load, attention recompute).
  2. Cross-transfer between models with different depths: validates shapes,
     layer selection, and that the pipeline runs on a genuine L_s != L_t pair.
"""

import json
import random

import pytest
import torch

from kv_mapper_fit.analysis import run_analysis
from kv_mapper_fit.capture import run_capture
from kv_mapper_fit.config import (
    PipelineConfig, analysis_path, diagnostics_path, mapper_path,
)
from kv_mapper_fit.diagnostics import run_diagnostics
from kv_mapper_fit.ridge import run_fit

VOCAB_SIZE = 128
SEQ_LEN = 64


def _make_tokenizer(path):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {"[UNK]": 0, "[PAD]": 1}
    for i in range(2, VOCAB_SIZE):
        vocab[f"w{i}"] = i
    tok = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok, unk_token="[UNK]", pad_token="[PAD]"
    )
    fast.save_pretrained(str(path))


def _make_model(path, num_layers, seed):
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(seed)
    cfg = LlamaConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=num_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=512,
        rope_theta=10000.0,
    )
    model = LlamaForCausalLM(cfg)
    model.save_pretrained(str(path))


def _make_corpus(path, num_tokens=6000, seed=0):
    rng = random.Random(seed)
    words = [f"w{rng.randrange(2, VOCAB_SIZE)}" for _ in range(num_tokens)]
    lines = [" ".join(words[i:i + 50]) for i in range(0, num_tokens, 50)]
    path.write_text("\n".join(lines))


@pytest.fixture(scope="module")
def assets(tmp_path_factory):
    root = tmp_path_factory.mktemp("assets")
    small, big = root / "small", root / "big"
    _make_model(small, num_layers=3, seed=1)
    _make_model(big, num_layers=5, seed=2)
    _make_tokenizer(small)
    _make_tokenizer(big)
    corpus = root / "corpus.txt"
    _make_corpus(corpus)
    return {"small": str(small), "big": str(big), "corpus": str(corpus)}


def _config(assets, workdir, source, target, top_k):
    return PipelineConfig(
        source_model=assets[source],
        target_model=assets[target],
        workdir=str(workdir),
        text_file=assets["corpus"],
        num_sequences=16,
        seq_len=SEQ_LEN,
        subsample_stride=2,
        probe_tokens=400,
        top_k=top_k,
        diag_sequences=2,
        diag_seq_len=48,
        batch_size=4,
        device="cpu",
        dtype="float32",
    )


def _run_all(cfg):
    cfg.save()
    run_capture(cfg)
    run_analysis(cfg)
    run_fit(cfg)
    return run_diagnostics(cfg)


def test_self_transfer_is_near_perfect(assets, tmp_path):
    cfg = _config(assets, tmp_path / "self", "small", "small", top_k=1)
    diag = _run_all(cfg)

    analysis = json.loads(analysis_path(cfg.resolved_workdir()).read_text())
    # the diagonal of the probe heatmap must be ~1 (each layer predicts itself)
    for layer, sel in enumerate(analysis["selected_layers"]):
        assert sel == [layer], f"self-transfer should select layer {layer}, got {sel}"
        assert analysis["r2_k"][layer][layer] > 0.99
        assert analysis["r2_v"][layer][layer] > 0.99

    assert diag["eval_r2_k"] > 0.99
    assert diag["eval_r2_v"] > 0.99
    assert diag["mean_attention_cosine"] > 0.995
    assert diag["tier"] == "tier1"


def test_cross_transfer_pipeline(assets, tmp_path):
    cfg = _config(assets, tmp_path / "cross", "small", "big", top_k=2)
    diag = _run_all(cfg)

    workdir = cfg.resolved_workdir()
    assert mapper_path(workdir).exists()
    assert diagnostics_path(workdir).exists()

    analysis = json.loads(analysis_path(workdir).read_text())
    assert len(analysis["selected_layers"]) == 5      # target layer count
    assert all(len(sel) == 2 for sel in analysis["selected_layers"])

    from kv_mapper_fit.mapper import KVCacheMapper
    mapper = KVCacheMapper.load(str(mapper_path(workdir)))
    assert mapper.meta["num_target_layers"] == 5
    assert mapper.meta["num_source_layers"] == 3
    # a random cross pair won't transfer well; just require sane values
    assert -1.0 <= diag["mean_attention_cosine"] <= 1.0
    assert diag["diag_sequences"] == 2


def test_mapper_shapes_roundtrip(assets, tmp_path):
    """map_stripped produces the target layout, and RoPE strip/apply round-trips."""
    from kv_mapper_fit.modeling import (
        CapturableModel, apply_rope, strip_rope,
    )

    cfg = _config(assets, tmp_path / "shapes", "small", "big", top_k=2)
    _run_all(cfg)

    from kv_mapper_fit.mapper import KVCacheMapper
    mapper = KVCacheMapper.load(str(mapper_path(cfg.resolved_workdir())))

    b, t, h, d = 1, 10, 2, 16
    model = CapturableModel(assets["small"], torch.device("cpu"), torch.float32)
    pos = torch.arange(t).unsqueeze(0)
    cos, sin = model.rope_cos_sin(pos)
    x = torch.randn(b, h, t, d)
    assert torch.allclose(strip_rope(apply_rope(x, cos, sin), cos, sin), x,
                          atol=1e-5)

    src_k = [torch.randn(b, h, t, d) for _ in range(3)]
    src_v = [torch.randn(b, h, t, d) for _ in range(3)]
    out_k, out_v = mapper.map_stripped(src_k, src_v, cos, sin)
    assert len(out_k) == 5 and len(out_v) == 5
    assert out_k[0].shape == (b, h, t, d)
    assert out_v[0].shape == (b, h, t, d)
