# System Architecture

## Overview

Modular monolith — no microservices. All modules are independently testable
Python classes/functions. The Streamlit UI calls pipeline modules directly
in-process (no separate API layer required).

```
Customer message + thread context
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│  Data Layer (src/data/)                                      │
│  ┌──────────┐  ┌────────────┐  ┌─────────┐  ┌──────────┐   │
│  │ loader.py│  │ threads.py │  │splits.py│  │ golden.py│   │
│  └──────────┘  └────────────┘  └─────────┘  └──────────┘   │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│  AI Layer (src/ai/)                                          │
│                                                              │
│  ┌──────────────────┐   ← llm_client.py (mock/openai/claude)│
│  │  classifier.py   │   Intent: KeywordBaseline | LLM        │
│  └────────┬─────────┘                                       │
│           │                                                  │
│  ┌────────▼─────────┐                                       │
│  │ src/retrieval/   │   Embedding index (numpy cosine)       │
│  │  index.py        │   ← Training split ONLY               │
│  └────────┬─────────┘                                       │
│           │                                                  │
│  ┌────────▼─────────┐                                       │
│  │  generator.py    │   Grounded reply generation            │
│  └────────┬─────────┘                                       │
│           │                                                  │
│  ┌────────▼─────────┐                                       │
│  │  validator.py    │   Grounding check (rules + LLM)        │
│  └────────┬─────────┘                                       │
│           │                                                  │
│  ┌────────▼─────────┐                                       │
│  │   router.py      │   AUTO_HANDLE | ESCALATE decision      │
│  └────────┬─────────┘                                       │
│           │                                                  │
│  ┌────────▼─────────┐                                       │
│  │  pipeline.py     │   Full orchestrated PipelineResult     │
│  └──────────────────┘                                       │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│  Evaluation Layer (src/eval/)                                │
│  ┌──────────┐  ┌────────┐  ┌──────────┐                    │
│  │metrics.py│  │judge.py│  │harness.py│                    │
│  └──────────┘  └────────┘  └──────────┘                    │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│  UI Layer (src/ui/)                                          │
│  Streamlit multi-page app — calls pipeline modules directly  │
│  ┌──────────┐  ┌─────────────┐  ┌───────────┐  ┌────────┐  │
│  │ inbox.py │  │eval_dashbrd │  │analytics.py│  │label   │  │
│  └──────────┘  └─────────────┘  └───────────┘  └────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Storage

```
data/
├── raw/                    ← Original CSV (gitignored)
├── cache/                  ← Parquet + JSONL splits (gitignored)
│   ├── brand_AmazonHelp_raw.parquet
│   ├── AmazonHelp_train.jsonl
│   ├── AmazonHelp_val.jsonl
│   ├── AmazonHelp_test.jsonl
│   ├── embeddings_*.npy    ← Cached embeddings
│   └── retrieval_index_AmazonHelp.pkl
├── gold/                   ← Human labels (labelled_by='human')
│   ├── candidate_pool.jsonl
│   └── golden_set.jsonl    ← THE authoritative evaluation set
└── feedback.db             ← SQLite human feedback log
```

## Key Design Decisions

1. **No microservices**: Streamlit calls modules directly. No FastAPI needed.
2. **Mock mode**: Entire pipeline runs without API keys — deterministic canned responses.
3. **Training-only retrieval**: Index built from train split only to prevent leakage.
4. **Uncalibrated confidence**: All confidence scores labeled as system signals in UI.
5. **SQLite for feedback**: Appropriate for single-machine demo; log-only, no retraining.
6. **Conversation-level splits**: Prevents the common message-level leakage failure.
