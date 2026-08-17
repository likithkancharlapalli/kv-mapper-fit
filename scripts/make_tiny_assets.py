"""Create tiny random models + corpus for an offline CLI demo/smoke run."""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from test_pipeline import _make_corpus, _make_model, _make_tokenizer  # noqa: E402

root = Path(sys.argv[1] if len(sys.argv) > 1 else "tiny_assets")
root.mkdir(parents=True, exist_ok=True)
_make_model(root / "small", num_layers=3, seed=1)
_make_model(root / "big", num_layers=5, seed=2)
_make_tokenizer(root / "small")
_make_tokenizer(root / "big")
_make_corpus(root / "corpus.txt")
print(f"tiny assets written to {root}")
