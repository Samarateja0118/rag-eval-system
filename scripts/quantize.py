"""Quantize a GGUF to Q8_0 with llama.cpp's quantizer (via llama-cpp-python).

The reranker ships as F16; Q8_0 halves its size. usage: python scripts/quantize.py IN.gguf OUT.gguf
"""

import ctypes
import sys

import llama_cpp


def main(src: str, dst: str) -> None:
    p = llama_cpp.llama_model_quantize_default_params()
    p.ftype = llama_cpp.LLAMA_FTYPE_MOSTLY_Q8_0
    rc = llama_cpp.llama_model_quantize(src.encode(), dst.encode(), ctypes.byref(p))
    if rc != 0:
        raise SystemExit(f"quantize failed rc={rc}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
