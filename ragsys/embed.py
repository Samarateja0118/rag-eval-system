"""Sentence embeddings with all-MiniLM-L6-v2 exported to ONNX (run on onnxruntime CPU).

The ONNX export is the one Chroma publishes (fetched by scripts/fetch_models.sh); we do the
mean-pooling + L2 normalisation ourselves, matching sentence-transformers' pipeline.
"""

from __future__ import annotations

import functools
from pathlib import Path

import numpy as np

MAX_LEN = 256  # all-MiniLM-L6-v2 positional limit used by sentence-transformers


class OnnxEmbedder:
    dim = 384

    def __init__(self, model_dir: str | Path, threads: int = 4, batch_size: int = 32):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        model_dir = Path(model_dir)
        self.tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.tok.enable_truncation(MAX_LEN)
        self.tok.enable_padding(pad_id=0, pad_token="[PAD]")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(
            str(model_dir / "model.onnx"), opts, providers=["CPUExecutionProvider"]
        )
        self.input_names = {i.name for i in self.sess.get_inputs()}
        self.batch_size = batch_size
        # untruncated counter for chunk sizing
        self._count_tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self._count_tok.no_truncation()
        self._count_tok.no_padding()

    @functools.lru_cache(maxsize=500_000)  # noqa: B019 - one embedder per process
    def count_tokens(self, text: str) -> int:
        return len(self._count_tok.encode(text, add_special_tokens=False).ids)

    def truncated(self, text: str) -> bool:
        return self.count_tokens(text) > MAX_LEN - 2

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        # sort by length so padding is minimal
        order = np.argsort([len(t) for t in texts])
        for s in range(0, len(texts), self.batch_size):
            idx = order[s : s + self.batch_size]
            enc = self.tok.encode_batch([texts[i] for i in idx])
            ids = np.array([e.ids for e in enc], dtype=np.int64)
            mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
            feeds = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in self.input_names:
                feeds["token_type_ids"] = np.zeros_like(ids)
            hidden = self.sess.run(None, feeds)[0]  # (b, t, 384)
            m = mask[..., None].astype(np.float32)
            pooled = (hidden * m).sum(1) / np.clip(m.sum(1), 1e-9, None)
            pooled /= np.linalg.norm(pooled, axis=1, keepdims=True) + 1e-12
            out[idx] = pooled
        return out
