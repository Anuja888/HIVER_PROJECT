"""
Phase 5-8 — Full AI Pipeline Orchestrator.

Runs the complete pipeline:
  customer message + thread context
    -> normalization (data layer)
    -> intent classification
    -> historical evidence retrieval
    -> reply generation
    -> grounding validation
    -> escalation routing
    -> structured PipelineResult

Every step is independently testable. The pipeline is stateless per call.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from src.ai.classifier import ClassificationResult, get_classifier
from src.ai.generator import GenerationOutput, generate_reply
from src.ai.router import RouterOutput, route
from src.ai.validator import ValidationResult, validate_grounding
from src.retrieval.index import RetrievalResult, get_index

log = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    # Input
    customer_message: str
    thread_context: list[dict] = field(default_factory=list)
    brand: str = "AmazonHelp"

    # Per-step outputs
    classification: Optional[ClassificationResult] = None
    retrieval: Optional[RetrievalResult] = None
    generation: Optional[GenerationOutput] = None
    validation: Optional[ValidationResult] = None
    routing: Optional[RouterOutput] = None

    # Timing
    latency_ms: float = 0.0
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "customer_message": self.customer_message,
            "brand": self.brand,
            "classification": self.classification.model_dump() if self.classification else None,
            "retrieval": {
                "top_similarity": self.retrieval.top_similarity if self.retrieval else 0,
                "has_sufficient_evidence": self.retrieval.has_sufficient_evidence if self.retrieval else False,
                "items": self.retrieval.to_dict()[:3] if self.retrieval else [],
            },
            "generation": self.generation.model_dump() if self.generation else None,
            "validation": self.validation.model_dump() if self.validation else None,
            "routing": self.routing.model_dump() if self.routing else None,
            "latency_ms": self.latency_ms,
            "error": self.error,
        }


def run_pipeline(
    customer_message: str,
    thread_context: Optional[list[dict]] = None,
    brand: str = "AmazonHelp",
    classifier_type: str = "auto",
    use_llm_grounding: bool = True,
) -> PipelineResult:
    """
    Run the full AI pipeline for a single customer message.

    Args:
        customer_message: The customer's message text
        thread_context: Previous messages in the thread (dicts with role/text)
        brand: Brand handle for retrieval index
        classifier_type: 'auto', 'llm_fewshot', or 'keyword_baseline'
        use_llm_grounding: Whether to use LLM for grounding check

    Returns:
        PipelineResult with all intermediate outputs
    """
    start = time.monotonic()
    result = PipelineResult(
        customer_message=customer_message,
        thread_context=thread_context or [],
        brand=brand,
    )

    try:
        # Step 1: Intent classification
        classifier = get_classifier(classifier_type)
        result.classification = classifier.predict(
            customer_message, context=thread_context
        )
        log.debug("Classification: %s (%.2f)", result.classification.intent,
                  result.classification.confidence)

        # Step 2: Evidence retrieval
        index = get_index(brand)
        result.retrieval = index.search(customer_message)
        log.debug("Retrieval: top_sim=%.3f, %d items",
                  result.retrieval.top_similarity, len(result.retrieval.items))

        # Step 3: Reply generation
        result.generation = generate_reply(
            customer_message=customer_message,
            intent=result.classification.intent,
            retrieval_result=result.retrieval,
            thread_context=thread_context,
        )

        # Step 4: Grounding validation
        result.validation = validate_grounding(
            reply=result.generation.reply,
            customer_message=customer_message,
            evidence_items=result.generation.evidence_items,
            use_llm=use_llm_grounding,
        )

        # Step 5: Escalation routing
        result.routing = route(
            classification=result.classification,
            retrieval=result.retrieval,
            validation=result.validation,
        )

    except Exception as e:
        log.error("Pipeline error: %s", e, exc_info=True)
        result.error = str(e)

    result.latency_ms = (time.monotonic() - start) * 1000
    return result
