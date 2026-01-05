"""
DSPy-compatible metrics for evaluating the LeadEvolverPipeline.

Training metric: Uses the static LLM Judge (no ground truth access)
Test metric: Direct comparison to human labels
"""

from typing import Any, Optional
from .judge import ClassifierJudge
from src.data_schema.blackboard import Blackboard


# Singleton judge instance to avoid repeated initialization
_judge_instance: Optional[ClassifierJudge] = None


def get_judge() -> ClassifierJudge:
    """Get or create the singleton LLM Judge instance."""
    global _judge_instance
    if _judge_instance is None:
        _judge_instance = ClassifierJudge()
    return _judge_instance


def safe_get(obj: Any, key: str, default: Any = None) -> Any:
    """
    Safely get a value from an object, handling both dict and attribute access.
    
    Args:
        obj: The object to get the value from (dict or object with attributes)
        key: The key/attribute name to access
        default: Default value if key not found
        
    Returns:
        The value, or default if not found
    """
    if obj is None:
        return default
    
    # Try dict-style access first
    if isinstance(obj, dict):
        return obj.get(key, default)
    
    # Try attribute access
    if hasattr(obj, key):
        return getattr(obj, key, default)
    
    # Try __getitem__ (for dict-like objects)
    try:
        return obj[key]
    except (KeyError, TypeError, IndexError):
        pass
    
    return default


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
    try:
        judge = get_judge()

        # Extract values using safe accessor
        lead_quality = safe_get(prediction, 'lead_quality', None)
        rationale = safe_get(prediction, 'rationale', None)
        blackboard_data = safe_get(prediction, 'blackboard', {})

        if lead_quality is None:
            print(f"DEBUG training_metric: No lead_quality found in prediction: {type(prediction)}")
            return 0.0

        # Convert blackboard to string
        if isinstance(blackboard_data, dict):
            blackboard_str = Blackboard.from_dict(blackboard_data).to_string()
        elif isinstance(blackboard_data, str):
            blackboard_str = blackboard_data
        elif hasattr(blackboard_data, 'to_string'):
            blackboard_str = blackboard_data.to_string()
        else:
            blackboard_str = str(blackboard_data) if blackboard_data else ""

        # Get judge's assessment (NO ground truth used here)
        judge_classification = judge.judge(
            lead_context=blackboard_str,
            proposed_classification=lead_quality,
            proposed_rationale=rationale
        )

        # Score based on alignment with judge
        score = compute_classification_score(lead_quality, judge_classification)
        return score
        
    except Exception as e:
        print(f"ERROR in training_metric: {e}")
        return 0.0


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
    try:
        # Extract ground truth from example
        ground_truth = safe_get(example, 'icp_match', None)

        if ground_truth is None:
            lead_username = safe_get(example, 'lead_username', 'unknown')
            print(f"WARNING: No ground truth found for example: {lead_username}")
            return 0.0

        # Extract prediction
        predicted = safe_get(prediction, 'lead_quality', None)
        if predicted is None:
            print(f"DEBUG test_set_metric: No lead_quality found in prediction: {type(prediction)}")
            return 0.0

        return compute_classification_score(predicted, ground_truth)
        
    except Exception as e:
        print(f"ERROR in test_set_metric: {e}")
        return 0.0
