# ML Model Integration Roadmap

Local model integrations for verification, retrieval, and knowledge extraction. All models are open-source, run on CPU, and require no API keys.

---

## Implemented

### HHEM-2.1-Open (Vectara) — Source Grounding
- **HuggingFace:** `vectara/hallucination_evaluation_model`
- **What:** Scores (source_text, finding_text) pairs for factual consistency
- **Size:** ~440MB (FLAN-T5-base, 0.1B params)
- **Speed:** <2s on CPU
- **Benchmarks:** AggreFact-SOTA 76.55%, RAGTruth-QA 74.28% (outperforms GPT-3.5-Turbo by 4-18%)
- **License:** Apache 2.0
- **Location:** `src/verification/hhem.py`
- **Pipeline position:** Runs before CoVe, enables early reject of ungrounded findings

---

## Next Up

### 1. DeBERTa-v3 NLI — Cross-Finding Contradiction Detection
- **HuggingFace:** `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`
- **What:** Best NLI model on HuggingFace (trained on 5 NLI datasets). Classifies pairs as entailment / neutral / contradiction.
- **Use case:** For every pair of findings on the same topic, check for contradictions. Flag pairs with >0.7 contradiction confidence in the knowledge graph.
- **Size:** ~1.7GB
- **Lighter alternative:** `MoritzLaurer/DeBERTa-v3-base-mnli-fever-docnli-ling-2c` (2-class: entailment vs contradiction, ~700MB)
- **Integration point:** `src/verification/` or `src/knowledge/graph.py` contradiction detection
- **Pipeline position:** After HHEM, before CoVe

### 2. AlignScore — Factual Consistency Ensemble
- **HuggingFace:** `yzha/AlignScore-large` (or `yzha/AlignScore-base`)
- **What:** Unified alignment scorer trained on 4.7M examples from diverse NLI/fact-checking datasets. State-of-the-art on SummaC benchmark.
- **Use case:** Ensemble with HHEM for higher-confidence grounding scores. Two models agreeing = much stronger signal.
- **Integration point:** `src/verification/hhem.py` → extend or create `src/verification/alignscore.py`
- **Pipeline position:** Alongside HHEM as ensemble

### 3. ClaimBuster — Check-Worthiness Gate
- **What:** Classifies sentences by check-worthiness (is this a verifiable claim or just opinion/fluff?)
- **Use case:** Pre-filter before running any verification. Skip verification entirely for non-factual findings (opinions, questions, suggestions).
- **Integration point:** `src/verification/pipeline.py` as first gate
- **Pipeline position:** Before HHEM (cheapest gate first)

---

## Retrieval & Reranking

### 4. FlashRank — Lightweight Reranker
- **What:** ~4-6MB reranker, no GPU, no torch dependency
- **Use case:** Replace or supplement current cross-encoder reranking with a faster option
- **Integration point:** `src/retrieval/reranker.py`

### 5. BGE-M3 — Multi-Granularity Embeddings
- **HuggingFace:** `BAAI/bge-m3`
- **What:** Dense + sparse + ColBERT embeddings in one model (same BAAI family as current BGE-large-en-v1.5)
- **Use case:** Upgrade from single-vector to multi-representation retrieval
- **Integration point:** `src/retrieval/hybrid.py`

---

## Knowledge Extraction

### 6. REBEL — End-to-End Relation Extraction
- **HuggingFace:** `Babelscape/rebel-large`
- **What:** Extracts (subject, relation, object) triples directly from text
- **Use case:** Automate knowledge graph construction from findings instead of relying on LLM extraction
- **Integration point:** `src/knowledge/graph.py`
- **Status:** Cut from initial plan (lacking enterprise validation), revisit later

### 7. GLiNER — Zero-Shot NER
- **What:** Named entity recognition with custom entity types at runtime (no fine-tuning)
- **Use case:** Extract domain-specific entities (drug names, gene names, company names) without pre-trained labels
- **Integration point:** `src/knowledge/graph.py` alongside spaCy NER
- **Status:** Cut from initial plan (too new), revisit later

---

## Target Verification Pipeline

```
Finding arrives
  → [1] ClaimBuster gate (is this even a verifiable claim?)
  → [2] HHEM grounding check (does it match the source?) ← IMPLEMENTED
  → [3] NLI contradiction check (does it conflict with other findings?)
  → [4] CoVe (LLM-based verification questions) ← EXISTS
  → [5] CRITIC (iterative critique, only if high-stakes) ← EXISTS
  → [6] Calibrated confidence score
```

Steps 1-3 are local model inference (<5s total on CPU). Only findings surviving initial gates hit expensive LLM calls. Estimated 30-50% verification cost reduction from gating.

---

## Selection Criteria

Models were evaluated against these requirements:
- **Enterprise validation:** 2M+ HuggingFace downloads or documented Fortune 500 usage
- **CPU-friendly:** Must run on CPU in reasonable time (<5s per inference)
- **License:** Apache 2.0 or MIT preferred
- **Size:** <2GB per model
- **No API keys:** Fully local inference
