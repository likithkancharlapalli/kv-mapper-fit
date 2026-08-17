"""kv-mapper-fit command line interface.

  kv-mapper-fit run      --source Qwen/Qwen3-14B --target Qwen/Qwen3-32B -w work/
  kv-mapper-fit capture  --source ... --target ... -w work/
  kv-mapper-fit analyze  -w work/
  kv-mapper-fit fit      -w work/
  kv-mapper-fit diagnose -w work/
"""

from __future__ import annotations

import argparse
import sys

from kv_mapper_fit.config import PipelineConfig


def _add_pair_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--source", required=True, help="source model (HF id or path)")
    p.add_argument("--target", required=True, help="target model (HF id or path)")
    p.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    p.add_argument("--dataset-config", default="sample-10BT")
    p.add_argument("--dataset-split", default="train")
    p.add_argument("--text-file", default=None,
                   help="local newline-delimited text corpus (overrides --dataset)")
    p.add_argument("--num-sequences", type=int, default=500)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--subsample-stride", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--trust-remote-code", action="store_true")


def _add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("-w", "--workdir", required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="auto")
    p.add_argument("--seed", type=int, default=0)


def _add_analyze_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--top-k", type=str, default="8",
                   help="source layers per target layer, or 'all'")
    p.add_argument("--probe-tokens", type=int, default=32768)


def _add_fit_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--ridge-lambda", type=float, default=0.01)


def _add_diag_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--diag-sequences", type=int, default=16)
    p.add_argument("--diag-seq-len", type=int, default=1024)


def _config_from_args(args) -> PipelineConfig:
    top_k = getattr(args, "top_k", "8")
    top_k = 0 if str(top_k).lower() == "all" else int(top_k)
    fields = dict(
        source_model=args.source,
        target_model=args.target,
        workdir=args.workdir,
        dataset=args.dataset,
        dataset_config=args.dataset_config,
        dataset_split=args.dataset_split,
        text_file=args.text_file,
        num_sequences=args.num_sequences,
        seq_len=args.seq_len,
        subsample_stride=args.subsample_stride,
        batch_size=args.batch_size,
        trust_remote_code=args.trust_remote_code,
        device=args.device,
        dtype=args.dtype,
        seed=args.seed,
        top_k=top_k,
    )
    for name in ("probe_tokens", "ridge_lambda", "diag_sequences",
                 "diag_seq_len"):
        if hasattr(args, name):
            fields[name] = getattr(args, name)
    return PipelineConfig(**fields)


def _loaded_config(args, **overrides) -> PipelineConfig:
    cfg = PipelineConfig.load(args.workdir)
    for name, value in overrides.items():
        setattr(cfg, name, value)
    cfg.device = args.device
    cfg.dtype = args.dtype
    return cfg


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="kv-mapper-fit",
        description=("Fit closed-form cross-model KV cache mappers "
                     "(arXiv:2608.03893)."),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("capture", help="extract calibration KV caches")
    _add_common_args(p)
    _add_pair_args(p)

    p = sub.add_parser("analyze", help="R^2 probe + top-k layer selection")
    _add_common_args(p)
    _add_analyze_args(p)

    p = sub.add_parser("fit", help="closed-form ridge fit + safetensors export")
    _add_common_args(p)
    _add_fit_args(p)

    p = sub.add_parser("diagnose", help="attention-output cosine + tier verdict")
    _add_common_args(p)
    _add_diag_args(p)

    p = sub.add_parser("run", help="full pipeline: capture, analyze, fit, diagnose")
    _add_common_args(p)
    _add_pair_args(p)
    _add_analyze_args(p)
    _add_fit_args(p)
    _add_diag_args(p)

    args = parser.parse_args(argv)

    if args.command == "capture":
        from kv_mapper_fit.capture import run_capture
        cfg = _config_from_args(args)
        cfg.save()
        run_capture(cfg)
    elif args.command == "analyze":
        from kv_mapper_fit.analysis import run_analysis
        top_k = 0 if str(args.top_k).lower() == "all" else int(args.top_k)
        cfg = _loaded_config(args, top_k=top_k, probe_tokens=args.probe_tokens)
        cfg.save()
        run_analysis(cfg)
    elif args.command == "fit":
        from kv_mapper_fit.ridge import run_fit
        cfg = _loaded_config(args, ridge_lambda=args.ridge_lambda)
        cfg.save()
        run_fit(cfg)
    elif args.command == "diagnose":
        from kv_mapper_fit.diagnostics import run_diagnostics
        cfg = _loaded_config(args, diag_sequences=args.diag_sequences,
                             diag_seq_len=args.diag_seq_len)
        cfg.save()
        run_diagnostics(cfg)
    elif args.command == "run":
        from kv_mapper_fit.analysis import run_analysis
        from kv_mapper_fit.capture import run_capture
        from kv_mapper_fit.diagnostics import run_diagnostics
        from kv_mapper_fit.ridge import run_fit
        cfg = _config_from_args(args)
        cfg.save()
        run_capture(cfg)
        run_analysis(cfg)
        run_fit(cfg)
        run_diagnostics(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
