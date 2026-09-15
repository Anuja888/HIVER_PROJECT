# Decision Log

10–15 non-obvious decisions made throughout this project, each with a one-line rationale.

---

1. **Brand: AmazonHelp chosen via data-driven weighted scoring** (not by intuition)  
   *Why*: Highest composite score (0.616) driven by 11k brand messages and 76 reconstructable multi-turn threads — 2× more than any competitor. Explainable, reproducible, and tied to a documented formula.

2. **Weighted scoring formula: reconstructable_threads weight = 0.25 (highest)**  
   *Why*: Multi-turn context is essential for grounding evaluation quality. A brand that only sends one-shot replies provides no evidence of resolution patterns. This is the most important data property for the downstream task.

3. **Conversation-level train/val/test split (not message-level)**  
   *Why*: Message-level splitting causes leakage — the model can see a customer's phrasing in training and recognize it in test. All messages from a thread must live in one split.

4. **Test split as source for golden evaluation set (never train/val)**  
   *Why*: Prevents any risk of the retrieval index or few-shot examples influencing which examples appear in the golden set. The test split has never been seen by any model component.

5. **Stratified sampling for golden set by intent bucket**  
   *Why*: Without stratification, the golden set would be dominated by `order_delivery_tracking` (the most common intent), making accuracy look better than it is and hiding per-intent failure modes.

6. **Thread truncation at MAX_CONTEXT_TURNS = 10**  
   *Why*: 95%+ of Amazon support threads in this dataset are ≤7 turns. Capping at 10 keeps LLM context short (cheaper/faster) while covering virtually all real conversations.

7. **Emoji preserved in message cleaning**  
   *Why*: Emoji carry sentiment signal (frustrated 😤, satisfied 😊) relevant to escalation decisions and tone of reply. Stripping them discards real information.

8. **LLM provider abstraction with mandatory mock mode**  
   *Why*: The assignment explicitly requires reproducibility without API cost. Mock mode uses deterministic hash-based canned responses so `make eval` produces consistent outputs regardless of API keys.

9. **confidence labeled "uncalibrated model signal" everywhere in UI**  
   *Why*: LLM self-reported confidence has no reliable probability interpretation without a calibration check (ECE/reliability diagram). Saying "probability" without calibration is misleading — the assignment rubric explicitly calls this out.

10. **Retrieval index capped at 5,000 training examples**  
    *Why*: Beyond ~3k examples, embedding a full 33k training split takes 10+ minutes on CPU. 5k examples is sufficient for retrieval quality while keeping index build time under 2 minutes. Diminishing returns above this size for nearest-neighbor retrieval.

11. **Simple numpy cosine similarity instead of FAISS**  
    *Why*: At 3k–5k vectors, brute-force dot product takes <10ms. FAISS adds installation complexity (BLAS dependencies, Windows build issues) for zero measurable speedup at this scale. "Boring reliable technology" preference stated in the prompt.

12. **Escalation threshold selected by validation-split sweep (not arbitrary)**  
    *Why*: Arbitrary thresholds are a common failure mode. The assignment explicitly requires threshold selection to be "by sweeping values against the validation split" — this is implemented in scripts/run_eval.py. Initial config.yaml values are placeholders.

13. **`account_security_compromise` intent is always escalated regardless of confidence**  
    *Why*: False-auto-handle on a compromised account can cause real financial harm. The risk is asymmetric — over-escalating security issues is cheap, under-escalating is catastrophic.

14. **Inter-annotator agreement target: second pass by same person or second person on 20–30 examples**  
    *Why*: The assignment requires measuring Cohen's kappa on a subset. With a single labeller (common in take-home projects), a second pass after a day produces a meaningful consistency check. If truly impractical, we document this explicitly rather than skipping the concept.

15. **Streamlit as UI framework, not React/Next.js**  
    *Why*: The assignment rubric rewards evaluation rigor, not frontend engineering. Streamlit lets the UI call pipeline modules directly in-process, eliminating a separate API layer and reducing reproducibility risk. A React app would add Webpack/npm/state-management complexity that could fail under the 15-minute reproducibility constraint.
