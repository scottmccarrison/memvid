=== STABLE BASELINE 2026-02-02T09:40:54-05:00 ===

Working state before vec/semantic search implementation.

## Features (Baseline - lex only)
- Lexical search (Tantivy/BM25) working
- Free tier: 1GB capacity
- CLI: memvid save/search/stats with --tag support

## Build Command (Baseline)
```bash
cargo build --release --features lex
```

## Restore to Baseline
```bash
cp ~/.local/bin/memvid.baseline ~/.local/bin/memvid
```

---

=== VEC/HYBRID UPDATE 2026-02-02T09:52:00-05:00 ===

Added semantic search with hybrid (lex + vec) ranking.

## Features (Current)
- Lexical search (Tantivy/BM25)
- Vector/semantic search (BGE-small embeddings via ONNX)
- Hybrid search using Reciprocal Rank Fusion (RRF)
- Embeddings auto-generated on save
- Free tier: 1GB capacity
- CLI: memvid save/search/stats/doctor/embed-all with --tag support

## Build Command (Current)
```bash
cargo build --release --features "lex,vec"
cp target/release/memvid ~/.local/bin/memvid
```

## Model Requirements
Text embedding model must be present at:
- `~/.cache/memvid/text-models/bge-small-en-v1.5.onnx`
- `~/.cache/memvid/text-models/bge-small-en-v1.5_tokenizer.json`

Download if missing:
```bash
mkdir -p ~/.cache/memvid/text-models
curl -L 'https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/main/onnx/model.onnx' \
  -o ~/.cache/memvid/text-models/bge-small-en-v1.5.onnx
curl -L 'https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/main/tokenizer.json' \
  -o ~/.cache/memvid/text-models/bge-small-en-v1.5_tokenizer.json
```

## Backfill Embeddings for Old Frames
If you have frames saved before enabling vec, generate embeddings for them:
```bash
memvid embed-all
```
This scans all frames, skips those with embeddings, and generates missing ones (~9/sec).

## Binary Sizes
- Baseline (lex only): ~15MB
- Current (lex + vec): ~44MB
