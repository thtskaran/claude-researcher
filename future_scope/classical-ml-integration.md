# Classical ML Integration for claude-researcher

*Leveraging traditional machine learning for speed, efficiency, and cost optimization*

---

## Why Classical ML?

LLMs are powerful but expensive, slow, and overkill for many tasks. Classical ML provides:

| Aspect | LLM | Classical ML |
|--------|-----|--------------|
| **Latency** | 500ms-5s | 10-50ms |
| **Cost** | $0.01-0.10 per call | Negligible (local) |
| **Memory** | 4-16GB+ | 50MB-500MB |
| **Accuracy (with training)** | Good | Often better |
| **Interpretability** | Black box | Explainable |

**Philosophy:** Use LLMs for reasoning and synthesis. Use classical ML for classification, ranking, filtering, and pattern matching.

---

## 1. Hybrid Retrieval (BM25 + Semantic)

### Problem
Pure semantic search misses exact keyword matches. Pure BM25 misses synonyms and context.

### Solution
Hybrid retrieval combining BM25 (lexical) with vector search (semantic).

**Performance data:**
- BM25 alone: 62% relevant docs in top 10
- Semantic alone: 71% relevant docs in top 10
- **Hybrid + reranking: 87% relevant docs in top 10**

Anthropic's Contextual Retrieval reduces failed retrievals by **49%** (67% with reranking).

### Implementation

```python
from rank_bm25 import BM25Okapi
import numpy as np
from sentence_transformers import SentenceTransformer

class HybridRetriever:
    def __init__(self, documents: list[str]):
        # BM25 index
        tokenized = [doc.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized)

        # Semantic index
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')  # 80MB, fast
        self.embeddings = self.encoder.encode(documents)
        self.documents = documents

    def search(self, query: str, top_k: int = 10, alpha: float = 0.5) -> list[tuple[int, float]]:
        """Hybrid search with Reciprocal Rank Fusion."""

        # BM25 scores
        bm25_scores = self.bm25.get_scores(query.lower().split())
        bm25_ranks = np.argsort(-bm25_scores)

        # Semantic scores
        query_emb = self.encoder.encode([query])[0]
        semantic_scores = np.dot(self.embeddings, query_emb)
        semantic_ranks = np.argsort(-semantic_scores)

        # Reciprocal Rank Fusion (k=60 works well universally)
        k = 60
        rrf_scores = {}
        for rank, doc_id in enumerate(bm25_ranks):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (k + rank + 1)
        for rank, doc_id in enumerate(semantic_ranks):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (k + rank + 1)

        # Sort by RRF score
        ranked = sorted(rrf_scores.items(), key=lambda x: -x[1])
        return ranked[:top_k]
```

### Where to Use
- Finding relevant past findings in knowledge graph
- Searching external memory store
- Deduplicating similar search queries

**Priority:** MUST-HAVE (immediate 15-30% recall improvement)

---

## 2. Learning to Rank (LambdaMART)

### Problem
LLM-based relevance scoring is slow and expensive. Can't rank 100+ documents per query.

### Solution
Train LambdaMART (gradient boosted trees) on relevance judgments to rank findings.

**Why LambdaMART:**
- Still SOTA baseline in 2024, often beats neural rankers
- XGBoost/LightGBM implementations are blazing fast
- Directly optimizes NDCG (what we actually care about)
- Inference: <10ms for 1000 documents

### Implementation

```python
import xgboost as xgb
import numpy as np

class FindingRanker:
    """Rank findings by relevance to research goal using LambdaMART."""

    def __init__(self):
        self.model = None
        self.feature_extractor = FeatureExtractor()

    def extract_features(self, query: str, finding: Finding) -> np.ndarray:
        """Extract ranking features for a query-finding pair."""
        return np.array([
            # Lexical features
            self.feature_extractor.bm25_score(query, finding.content),
            self.feature_extractor.query_term_coverage(query, finding.content),
            self.feature_extractor.title_match_score(query, finding.source_title),

            # Semantic features
            self.feature_extractor.embedding_similarity(query, finding.content),

            # Quality features
            finding.confidence,
            finding.credibility_score,
            self.feature_extractor.source_authority(finding.source_url),

            # Freshness
            self.feature_extractor.recency_score(finding.created_at),

            # Finding type signal
            1.0 if finding.finding_type == 'fact' else 0.5,

            # Length features
            len(finding.content),
            len(finding.content.split()),
        ])

    def train(self, training_data: list[tuple[str, list[Finding], list[int]]]):
        """Train on (query, findings, relevance_labels) tuples."""
        X, y, groups = [], [], []

        for query, findings, labels in training_data:
            group_size = len(findings)
            for finding, label in zip(findings, labels):
                X.append(self.extract_features(query, finding))
                y.append(label)
            groups.append(group_size)

        dtrain = xgb.DMatrix(np.array(X), label=np.array(y))
        dtrain.set_group(groups)

        params = {
            'objective': 'rank:ndcg',
            'eval_metric': 'ndcg@10',
            'eta': 0.1,
            'max_depth': 6,
            'subsample': 0.8,
        }

        self.model = xgb.train(params, dtrain, num_boost_round=100)

    def rank(self, query: str, findings: list[Finding]) -> list[Finding]:
        """Rank findings by predicted relevance."""
        if not findings:
            return []

        X = np.array([self.extract_features(query, f) for f in findings])
        dtest = xgb.DMatrix(X)
        scores = self.model.predict(dtest)

        ranked_indices = np.argsort(-scores)
        return [findings[i] for i in ranked_indices]
```

### Training Data
- User feedback on report quality (implicit)
- Manager's critique decisions (which findings were kept/rejected)
- Click-through on interactive reports

**Priority:** SHOULD-HAVE (enables fast ranking of large finding sets)

---

## 3. Fast Entity Extraction (spaCy)

### Problem
LLM-based entity extraction for knowledge graph is slow (~1s per finding) and expensive.

### Solution
Use spaCy for fast NER, fall back to LLM only for complex/ambiguous cases.

**Speed comparison:**
- spaCy: 10,000+ docs/second
- LLM: ~1 doc/second
- **1000x faster**

**Memory:**
- spaCy en_core_web_sm: 12MB
- spaCy en_core_web_trf: 400MB (transformer-based, more accurate)
- LLM: 4-16GB

### Implementation

```python
import spacy
from dataclasses import dataclass

@dataclass
class ExtractedEntity:
    text: str
    label: str
    confidence: float
    start: int
    end: int

class FastEntityExtractor:
    """Fast entity extraction using spaCy, LLM fallback for complex cases."""

    LABEL_MAP = {
        'PERSON': 'PERSON',
        'ORG': 'ORGANIZATION',
        'GPE': 'LOCATION',
        'LOC': 'LOCATION',
        'DATE': 'DATE',
        'MONEY': 'METRIC',
        'PERCENT': 'METRIC',
        'PRODUCT': 'TECHNOLOGY',
        'WORK_OF_ART': 'SOURCE',
    }

    def __init__(self, model: str = "en_core_web_sm"):
        self.nlp = spacy.load(model)
        self.llm_callback = None

    def extract(self, text: str, use_llm_fallback: bool = True) -> list[ExtractedEntity]:
        """Extract entities from text."""
        doc = self.nlp(text)

        entities = []
        for ent in doc.ents:
            if ent.label_ in self.LABEL_MAP:
                entities.append(ExtractedEntity(
                    text=ent.text,
                    label=self.LABEL_MAP[ent.label_],
                    confidence=0.85,
                    start=ent.start_char,
                    end=ent.end_char,
                ))

        if use_llm_fallback and self._needs_llm_extraction(text, entities):
            llm_entities = await self._llm_extract(text)
            entities = self._merge_entities(entities, llm_entities)

        return entities

    def _needs_llm_extraction(self, text: str, entities: list[ExtractedEntity]) -> bool:
        """Heuristic: use LLM if spaCy found very few entities in long text."""
        words = len(text.split())
        entity_density = len(entities) / max(words, 1)

        if words > 100 and entity_density < 0.02:
            return True

        technical_signals = ['algorithm', 'model', 'framework', 'architecture',
                           'technique', 'method', 'approach', 'system']
        if any(signal in text.lower() for signal in technical_signals):
            if not any(e.label == 'TECHNOLOGY' for e in entities):
                return True

        return False

    def extract_batch(self, texts: list[str]) -> list[list[ExtractedEntity]]:
        """Batch extraction for efficiency."""
        results = []
        for doc in self.nlp.pipe(texts, batch_size=50):
            entities = []
            for ent in doc.ents:
                if ent.label_ in self.LABEL_MAP:
                    entities.append(ExtractedEntity(
                        text=ent.text,
                        label=self.LABEL_MAP[ent.label_],
                        confidence=0.85,
                        start=ent.start_char,
                        end=ent.end_char,
                    ))
            results.append(entities)
        return results
```

**Priority:** MUST-HAVE (10-100x speedup for KG construction)

---

## 4. Document Deduplication (MinHash LSH)

### Problem
Research often finds the same information from multiple sources. Need to deduplicate without expensive pairwise comparisons.

### Solution
MinHash with Locality Sensitive Hashing - O(n) instead of O(n²).

**Performance:**
- 10K docs vs 5M historical: 50 billion comparisons naively
- With MinHash LSH: ~10M comparisons (5000x reduction)
- Latency: <100ms for dedup check

### Implementation

```python
from datasketch import MinHash, MinHashLSH

class FindingDeduplicator:
    """Fast near-duplicate detection using MinHash LSH."""

    def __init__(self, threshold: float = 0.7, num_perm: int = 128):
        self.threshold = threshold
        self.num_perm = num_perm
        self.lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        self.minhashes = {}

    def _text_to_shingles(self, text: str, k: int = 3) -> set:
        """Convert text to k-shingles."""
        text = text.lower()
        words = text.split()
        if len(words) < k:
            return set(words)
        return set(' '.join(words[i:i+k]) for i in range(len(words) - k + 1))

    def _create_minhash(self, text: str) -> MinHash:
        """Create MinHash signature for text."""
        m = MinHash(num_perm=self.num_perm)
        for shingle in self._text_to_shingles(text):
            m.update(shingle.encode('utf8'))
        return m

    def add_finding(self, finding_id: str, content: str) -> list[str]:
        """Add finding and return IDs of near-duplicates found."""
        minhash = self._create_minhash(content)
        duplicates = self.lsh.query(minhash)
        self.lsh.insert(finding_id, minhash)
        self.minhashes[finding_id] = minhash
        return duplicates

    def is_duplicate(self, content: str) -> tuple[bool, str]:
        """Check if content is a near-duplicate."""
        minhash = self._create_minhash(content)
        duplicates = self.lsh.query(minhash)

        if not duplicates:
            return False, None

        best_id = None
        best_similarity = 0
        for dup_id in duplicates:
            if dup_id in self.minhashes:
                similarity = minhash.jaccard(self.minhashes[dup_id])
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_id = dup_id

        return True, best_id
```

**Priority:** MUST-HAVE (prevents redundant findings, saves LLM costs)

---

## 5. Topic Modeling (BERTopic)

### Problem
Need to automatically discover themes/topics in findings for report organization.

### Solution
BERTopic combines BERT embeddings with HDBSCAN clustering and c-TF-IDF topic representation.

**Why BERTopic over LDA:**
- Understands semantic meaning, not just word frequency
- No need to specify number of topics upfront
- Handles short texts well

### Implementation

```python
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

class FindingTopicModeler:
    """Discover topics in research findings using BERTopic."""

    def __init__(self):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.topic_model = BERTopic(
            embedding_model=self.embedding_model,
            min_topic_size=3,
            nr_topics='auto',
            calculate_probabilities=True,
            verbose=False,
        )
        self.is_fitted = False

    def fit(self, findings: list[Finding]) -> dict:
        """Fit topic model on findings."""
        texts = [f.content for f in findings]

        if len(texts) < 5:
            return {'topics': [{'id': 0, 'name': 'General', 'keywords': []}],
                    'assignments': [0] * len(findings)}

        topics, probs = self.topic_model.fit_transform(texts)
        self.is_fitted = True

        topic_info = self.topic_model.get_topic_info()

        result = {'topics': [], 'assignments': topics}

        for _, row in topic_info.iterrows():
            if row['Topic'] != -1:
                topic_words = self.topic_model.get_topic(row['Topic'])
                result['topics'].append({
                    'id': row['Topic'],
                    'name': row['Name'],
                    'keywords': [word for word, _ in topic_words[:5]],
                    'count': row['Count'],
                })

        return result

    def suggest_report_sections(self) -> list[dict]:
        """Suggest report sections based on discovered topics."""
        if not self.is_fitted:
            return []

        topic_info = self.topic_model.get_topic_info()
        sections = []

        for _, row in topic_info.iterrows():
            if row['Topic'] != -1 and row['Count'] >= 3:
                topic_words = self.topic_model.get_topic(row['Topic'])
                keywords = [word for word, _ in topic_words[:3]]
                sections.append({
                    'title': ' & '.join([w.title() for w in keywords]),
                    'topic_id': row['Topic'],
                    'finding_count': row['Count'],
                })

        sections.sort(key=lambda x: -x['finding_count'])
        return sections
```

**Priority:** SHOULD-HAVE (improves report organization)

---

## 6. Text Classification (Distilled Models)

### Problem
Need to classify findings by type quickly without LLM calls.

### Solution
Train lightweight classifiers on LLM-labeled data (knowledge distillation).

```python
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

class FindingClassifier:
    """Fast classification using distilled models."""

    FINDING_TYPES = ['fact', 'insight', 'connection', 'question', 'contradiction', 'source']

    def __init__(self):
        self.classifier = None

    def train(self, texts: list[str], labels: list[str]):
        """Train finding type classifier."""
        self.classifier = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            ('clf', LogisticRegression(max_iter=1000, class_weight='balanced'))
        ])
        self.classifier.fit(texts, labels)

    def predict(self, text: str) -> tuple[str, float]:
        """Predict finding type with confidence."""
        if not self.classifier:
            return 'fact', 0.5

        probs = self.classifier.predict_proba([text])[0]
        pred_idx = np.argmax(probs)
        return self.classifier.classes_[pred_idx], probs[pred_idx]
```

**Priority:** SHOULD-HAVE (reduces LLM calls by 80%+)

---

## 7. Novelty Detection (Isolation Forest)

### Problem
Identify findings that are unusual/novel compared to existing knowledge.

### Solution
Isolation Forest - novel findings are "anomalies" relative to baseline.

```python
from sklearn.ensemble import IsolationForest
from sentence_transformers import SentenceTransformer

class NoveltyDetector:
    """Detect novel findings relative to existing knowledge."""

    def __init__(self):
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.model = IsolationForest(contamination=0.1, random_state=42)
        self.baseline_embeddings = None

    def fit_baseline(self, baseline_texts: list[str]):
        """Fit on existing knowledge base."""
        self.baseline_embeddings = self.encoder.encode(baseline_texts)
        self.model.fit(self.baseline_embeddings)

    def score_novelty(self, text: str) -> float:
        """Score how novel a finding is (0-1)."""
        if self.baseline_embeddings is None:
            return 0.5

        embedding = self.encoder.encode([text])
        raw_score = self.model.decision_function(embedding)[0]
        novelty = 1 - (raw_score + 0.5)
        return float(np.clip(novelty, 0, 1))
```

**Priority:** NICE-TO-HAVE (highlight surprising findings)

---

## Integration Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Finding Pipeline                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Raw Text ──→ [spaCy NER] ──→ Entities (fast, 10ms)             │
│      │              │                                            │
│      │              └──→ Complex? ──→ [LLM NER] (fallback)      │
│      │                                                           │
│      └──→ [MinHash LSH] ──→ Is Duplicate? ──→ Skip/Merge        │
│                  │                                               │
│                  └──→ [TF-IDF Classifier] ──→ Finding Type      │
│                              │                                   │
│                              └──→ Low confidence? ──→ [LLM]     │
│                                                                  │
│  Findings ──→ [BM25 + Semantic] ──→ Relevant to goal?           │
│      │                                                           │
│      └──→ [LambdaMART Ranker] ──→ Ordered by relevance          │
│                                                                  │
│  All Findings ──→ [BERTopic] ──→ Topic clusters                 │
│                       │                                          │
│                       └──→ Report section suggestions            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Cost/Speed Impact

| Component | LLM Cost | Classical ML Cost | Speedup |
|-----------|----------|-------------------|---------|
| Entity extraction | $0.01/finding | ~$0 | 100x |
| Deduplication | $0.02/comparison | ~$0 | 1000x |
| Type classification | $0.005/finding | ~$0 | 50x |
| Relevance ranking | $0.01/finding | ~$0 | 100x |
| Topic modeling | $0.05/batch | ~$0 | 20x |

**Estimated savings:** 70-80% reduction in LLM API costs.

---

## Implementation Priority

### Phase 1 (Week 1-2)
1. **Hybrid Retrieval** - BM25 + semantic search
2. **Fast NER** - spaCy with LLM fallback
3. **Deduplication** - MinHash LSH

### Phase 2 (Week 3-4)
4. **Finding Classifier** - Distilled type models
5. **LambdaMART Ranker** - Train on feedback

### Phase 3 (Month 2)
6. **BERTopic** - Auto report structuring
7. **Novelty Detection** - Highlight surprises

---

## Dependencies

```
# requirements.txt additions
rank-bm25>=0.2.2
sentence-transformers>=2.2.0
spacy>=3.5.0
datasketch>=1.6.0
bertopic>=0.15.0
xgboost>=2.0.0
scikit-learn>=1.3.0
```

**Total footprint:** ~500MB models + ~200MB libraries

---

## Sources

- [Hybrid Search Performance](https://www.fuzzylabs.ai/blog-post/improving-rag-performance-hybrid-search)
- [Anthropic Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval)
- [LambdaMART Explained](https://www.shaped.ai/blog/lambdamart-explained-the-workhorse-of-learning-to-rank)
- [spaCy vs LLM for NER](https://explosion.ai/blog/against-llm-maximalism)
- [MinHash LSH](https://milvus.io/blog/minhash-lsh-in-milvus-the-secret-weapon-for-fighting-duplicates-in-llm-training-data.md)
- [BERTopic](https://bertopic.com/)
