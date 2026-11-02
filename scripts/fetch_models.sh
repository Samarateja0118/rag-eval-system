#!/usr/bin/env bash
# Fetch all model weights into ./models (or $RAG_MODEL_DIR). Checksums pinned to what the results used.
set -euo pipefail
DIR="${RAG_MODEL_DIR:-models}"
mkdir -p "$DIR"

# all-MiniLM-L6-v2 ONNX export published by Chroma
if [ ! -f "$DIR/all-MiniLM-L6-v2/model.onnx" ]; then
  curl -fsSL -o /tmp/minilm.tar.gz https://chroma-onnx-models.s3.amazonaws.com/all-MiniLM-L6-v2/onnx.tar.gz
  echo "913d7300ceae3b2dbc2c50d1de4baacab4be7b9380491c27fab7418616a16ec3  /tmp/minilm.tar.gz" | sha256sum -c -
  tar -xzf /tmp/minilm.tar.gz -C "$DIR" && mv "$DIR/onnx" "$DIR/all-MiniLM-L6-v2" && rm /tmp/minilm.tar.gz
fi

# GGUF weights from Docker Hub's ai/ model catalog
python scripts/pull_model.py ai/qwen3-reranker 0.6B "$DIR/qwen3-reranker-0.6b.gguf" \
  9c9c23f7de6dce710fe77bb698659d06179d42075b13834af260c3f2145af662
# Q8_0 reranker: ~2x faster than F16 on CPU (measured 236 vs 120 prompt tok/s on 4 cores)
[ -f "$DIR/qwen3-reranker-0.6b-q8_0.gguf" ] || \
  python scripts/quantize.py "$DIR/qwen3-reranker-0.6b.gguf" "$DIR/qwen3-reranker-0.6b-q8_0.gguf"
python scripts/pull_model.py ai/qwen3 4b-instruct-2507-q4_K_M "$DIR/qwen3-4b-instruct-2507-q4_k_m.gguf" \
  3605803b982cb64aead44f6c1b2ae36e3acdb41d8e46c8a94c6533bc4c67e597
