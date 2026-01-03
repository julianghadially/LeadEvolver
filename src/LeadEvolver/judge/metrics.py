"""
DSPy-compatible metrics for evaluating the LeadEvolverPipeline.

Training metric: Uses the static LLM Judge (no ground truth access)
Test metric: Direct comparison to human labels
"""

from typing import Any, Optional
from .llm_judge import LLMJudge
from src.data_schema.blackboard import Blackboard


# Singleton judge instance to avoid repeated initialization
_judge_instance: Optional[LLMJudge] = None


def get_judge() -> LLMJudge:
    """Get or create the singleton LLM Judge instance."""
    global _judge_instance
    if _judge_instance is None:
        _judge_instance = LLMJudge()
    return _judge_instance


def compute_classification_score(predicted: str, expected: str) -> float:
    """
    Compute the score for a classification prediction.

    All inputs should be in snake_case format: strong_fit, weak_fit, not_a_fit

    Scoring:
        - 1.0: Exact match
        - 0.5: Confusion between strong_fit and weak_fit (or vice versa)
        - 0.0: Any other mismatch (e.g., fit vs not_a_fit)

    Args:
        predicted: The predicted classification (strong_fit, weak_fit, not_a_fit)
        expected: The expected classification (from judge or human label)

    Returns:
        Score between 0.0 and 1.0
    """
    # Exact match
    if predicted == expected:
        return 1.0

    # Partial credit for strong_fit <-> weak_fit confusion
    fit_classes = {"strong_fit", "weak_fit"}
    if predicted in fit_classes and expected in fit_classes:
        return 0.5

    # No credit for other mismatches (e.g., classifying not_a_fit as fit)
    return 0.0


def training_metric(
    example: Any,
    prediction: Any,
    trace: Optional[Any] = None,
    pred_name: Optional[str] = None,
    pred_trace: Optional[Any] = None
) -> float:
    """
    DSPy-compatible metric for TRAINING set evaluation.

    Uses the static LLM Judge to evaluate predictions.
    The judge has access to few-shot examples but NOT to ground truth
    of training examples.

    Args:
        example: DSPy example containing lead context
        prediction: Pipeline prediction with 'lead_quality' and 'blackboard' fields
        trace: Optional trace object (for DSPy compatibility)
        pred_name: Optional predictor name (for GEPA compatibility)
        pred_trace: Optional predictor trace (for GEPA compatibility)

    Returns:
        Score between 0.0 and 1.0
    """
    judge = get_judge()

    # Extract context from prediction (the researched context)
    blackboard = Blackboard.from_dict(prediction.get('blackboard', '')).to_string()
    lead_quality = prediction.get('lead_quality', None)
    rationale = prediction.get('rationale', None)

    if lead_quality is None:
        return 0.0

    # Get judge's assessment (NO ground truth used here)
    judge_classification = judge.judge(
        lead_context=blackboard,
        proposed_classification=lead_quality,
        proposed_rationale=rationale
    )

    # Score based on alignment with judge
    return compute_classification_score(lead_quality, judge_classification)


def test_set_metric(
    example: Any,
    prediction: Any,
    trace: Optional[Any] = None,
    pred_name: Optional[str] = None,
    pred_trace: Optional[Any] = None
) -> float:
    """
    DSPy-compatible metric for TEST set evaluation.

    Direct comparison to human labels. Only use this for test set
    where we want to measure actual accuracy against ground truth.

    Args:
        example: DSPy example containing ground truth in 'icp_match' field
        prediction: Pipeline prediction with 'lead_quality' field
        trace: Optional trace object (for DSPy compatibility)
        pred_name: Optional predictor name (for GEPA compatibility)
        pred_trace: Optional predictor trace (for GEPA compatibility)

    Returns:
        Score between 0.0 and 1.0
    """
    # Extract ground truth from example
    
    ground_truth = example.icp_match

    if ground_truth is None:
        print(f"WARNING: No ground truth found for example: {example.lead_username}")
        return 0.0

    # Extract prediction
    predicted = prediction['lead_quality']
    if predicted is None:
        return 0.0

    return compute_classification_score(predicted, ground_truth)
