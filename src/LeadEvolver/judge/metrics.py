"""
DSPy-compatible metrics for evaluating the LeadEvolverPipeline.

Training metric: Uses the static LLM Judge (no ground truth access)
Test metric: Direct comparison to human labels
"""

from typing import Any, Optional
from .llm_judge import LLMJudge


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
    trace: Optional[Any] = None
) -> float:
    """
    DSPy-compatible metric for TRAINING set evaluation.

    Uses the static LLM Judge to evaluate predictions.
    The judge has access to few-shot examples but NOT to ground truth
    of training examples.

    Args:
        example: DSPy example containing lead context
        prediction: Pipeline prediction with 'lead_quality' and 'blackboard' fields
        trace: Optional trace object (unused, for DSPy compatibility)

    Returns:
        Score between 0.0 and 1.0
    """
    judge = get_judge()

    # Extract context from prediction (the researched context)
    if hasattr(prediction, 'blackboard'):
        lead_context = prediction.blackboard
    elif isinstance(prediction, dict):
        lead_context = prediction.get('blackboard', '')
    else:
        lead_context = getattr(prediction, 'blackboard', '')

    # If no blackboard, try to use example context
    if not lead_context:
        if hasattr(example, 'context'):
            lead_context = example.context
        elif isinstance(example, dict):
            lead_context = example.get('context', '')

    # Extract predicted classification
    if hasattr(prediction, 'lead_quality'):
        predicted = prediction.lead_quality
    elif isinstance(prediction, dict):
        predicted = prediction.get('lead_quality')
    else:
        predicted = getattr(prediction, 'lead_quality', None)

    if predicted is None or not lead_context:
        return 0.0

    # Get rationale if available
    if hasattr(prediction, 'rationale'):
        rationale = prediction.rationale
    elif isinstance(prediction, dict):
        rationale = prediction.get('rationale')
    else:
        rationale = None

    # Get judge's assessment (NO ground truth used here)
    judge_classification = judge.judge(
        lead_context=lead_context,
        proposed_classification=predicted,
        proposed_rationale=rationale
    )

    # Score based on alignment with judge
    return compute_classification_score(predicted, judge_classification)


def test_set_metric(
    example: Any,
    prediction: Any,
    trace: Optional[Any] = None
) -> float:
    """
    DSPy-compatible metric for TEST set evaluation.

    Direct comparison to human labels. Only use this for test set
    where we want to measure actual accuracy against ground truth.

    Args:
        example: DSPy example containing ground truth in 'icp_match' field
        prediction: Pipeline prediction with 'lead_quality' field
        trace: Optional trace object (unused, for DSPy compatibility)

    Returns:
        Score between 0.0 and 1.0
    """
    # Extract ground truth from example
    if hasattr(example, 'icp_match'):
        ground_truth = example.icp_match
    elif isinstance(example, dict):
        ground_truth = example.get('icp_match')
    else:
        ground_truth = getattr(example, 'icp_match', None)

    if ground_truth is None:
        return 0.0

    # Extract prediction
    if hasattr(prediction, 'lead_quality'):
        predicted = prediction.lead_quality
    elif isinstance(prediction, dict):
        predicted = prediction.get('lead_quality')
    else:
        predicted = getattr(prediction, 'lead_quality', None)

    if predicted is None:
        return 0.0

    return compute_classification_score(predicted, ground_truth)
