Now I have the full picture. Here's the comprehensive diagnosis:                                                                                                                                   
                                                                                                                                                                                                     
  ---                                                                                                                                                                                                
  Diagnosis: Verification Pipeline for Session #33f0460
                                                                                                                                                                                                     
  Session Stats                                                                                                                                                                                      
                                                                                                                                                                                                     
  - 143 findings total: 66 verified, 69 rejected, 8 flagged                                                                                                                                          
  - Topic: "Which AI Coding Tools Actually Ship Production Code?"                                                                                                                                    
                                                                                                                                                                                                     
  ---
  5 Root Causes Identified

  1. HHEM is mass-rejecting 48% of findings on bogus grounds (THE BIG ONE)

  All 69 rejected findings share an identical fingerprint:
  - Exactly -0.15 confidence drop
  - 0ms verification time
  - Empty questions, contradictions, corrections
  - No error recorded

  This matches the early-reject path in pipeline.py:101-111:

  if 0 <= hhem_score < self.config.hhem_reject_threshold:  # threshold = 0.25
      result = VerificationResult(
          verified_confidence=max(0.0, finding.confidence - 0.15),
          verification_status=VerificationStatus.REJECTED,
      )

  Why HHEM scores are so low: The intern passes source_snippet (a search result snippet, ~100-200 characters) as source_content (intern.py:701,811). HHEM expects substantial premise text to
  evaluate grounding. A tiny snippet like "Cursor AI is an AI-powered code editor that helps developers write code faster..." doesn't contain enough detail to support a multi-sentence finding, so
  HHEM scores it < 0.25 and rejects it instantly — before CoVe even runs.

  Per the https://huggingface.co/vectara/hallucination_evaluation_model, the model evaluates whether a hypothesis is evidenced by a premise. Short snippets are simply insufficient premises.

  2. CoVe violates the independence principle from the original paper

  The https://arxiv.org/abs/2309.11495 explicitly states step 3 should answer questions "independently so the answers are not biased by other responses." But our implementation in cove.py:367-379
  does the opposite:

  prompt = (
      "Answer this verification question based on your knowledge, "
      "then determine whether your answer supports or contradicts "
      "the original claim.\n\n"
      f"ORIGINAL CLAIM: {original_finding}\n\n"  # ← BIAS
      f"VERIFICATION QUESTION: {q.question}\n"
  )

  By showing the original claim, the LLM is anchored to it — this is exactly the https://learnprompting.org/docs/advanced/self_criticism/chain_of_verification documented in CoVe implementations.
  The result: consistency scores cluster at 0.5 (neutral), providing no real signal.

  3. Knowledge Graph integration is completely dead

  Despite the KG having 426 entities and 371 relations for this session:
  - kg_support_score = 0.0 for all 143 findings
  - kg_entity_matches = 0 for all 143 findings
  - kg_supporting_relations = 0 for all 143 findings

  Why: _quick_entity_extract() in graph.py uses a narrow regex fallback ([A-Z][a-z]+) that misses domain terms like "cursor", "copilot", "machine learning". Then entity_by_name lookup fails because
   extracted names don't match KG entity names (case/format differences). Result: the KG is rich with data but verification can never see it.

  4. CRITIC never runs

  0 CRITIC iterations across all 143 findings. CRITIC only exists in the batch verification path (pipeline.py:307-327), but interns use streaming verification (verify_intern_finding). Batch
  verification is never triggered for this session — it requires an explicit verify_batch() call from the manager during synthesis, which apparently didn't happen or wasn't connected.

  5. Silent zeroes destroy transparency

  Every metric defaults to 0 or 0.0 in both the dataclass (models.py:64-69) and the DB schema (DEFAULT 0.0). There's no way to distinguish "not computed" from "computed as zero." The UI dutifully
  displays these zeroes, making the system look broken even where it's just not running.

  ---
  The Cascade Effect

  Finding arrives at intern
    → HHEM gets a tiny snippet as "source" → scores < 0.25 → EARLY REJECT (48% of findings)
    → For survivors: CoVe runs but shows the original claim → neutral 0.5 scores
    → KG lookup fails (entity mismatch) → 0 support everywhere
    → CRITIC never runs (streaming-only path)
    → Everything stored with silent 0 defaults → UI shows zeroes

  ---
  Fixes Needed

  Want me to implement these? Here's what needs to change:

  1. HHEM fix (pipeline.py): Skip HHEM early-reject when source_content is just a snippet (< 500 chars). Only use HHEM when full page content is available. Alternatively, pass the scraped page
  markdown instead of the search snippet.
  2. CoVe independence fix (cove.py): Split into two-step factored verification — first answer the question independently (without seeing the claim), then compare against the claim in a separate
  step. This is what the paper actually prescribes.
  3. KG entity matching fix (graph.py): Use fuzzy/normalized matching in get_kg_support_score() — lowercase both sides, use substring matching or the existing entity aliases, and improve
  _quick_entity_extract() to use spaCy NER instead of regex.
  4. Run batch verification (manager.py): Ensure verify_batch() is called during synthesis so CRITIC can run on high-stakes findings.
  5. NULL vs zero (models.py + db.py): Use None/NULL for "not computed" fields, and have the UI show "—" instead of "0" for uncomputed metrics.

  Sources:
  - https://arxiv.org/abs/2309.11495
  - https://learnprompting.org/docs/advanced/self_criticism/chain_of_verification
  - https://huggingface.co/vectara/hallucination_evaluation_model
  - https://www.vectara.com/blog/hallucination-detection-commercial-vs-open-source-a-deep-dive
