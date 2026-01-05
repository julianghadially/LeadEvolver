"""
DSPy-compatible metrics for evaluating the LeadProfilerPipeline.

Training metric: Uses the ProfileJudge to score profile quality (0-100)
"""

from typing import Any, Optional
from .judge import ProfileJudge
from src.data_schema.blackboard import Blackboard


# Singleton judge instance to avoid repeated initialization
_judge_instance: Optional[ProfileJudge] = None


def get_judge() -> ProfileJudge:
    """Get or create the singleton Profile Judge instance."""
    global _judge_instance
    if _judge_instance is None:
        _judge_instance = ProfileJudge()
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


def training_metric(
    example: Any,
    prediction: Any,
    trace: Optional[Any] = None,
    pred_name: Optional[str] = None,
    pred_trace: Optional[Any] = None
) -> float:
    """
    DSPy-compatible metric for TRAINING set evaluation.

    Uses the ProfileJudge to score profile quality on a 0-100 scale,
    then normalizes to 0.0-1.0 for DSPy compatibility.

    Args:
        example: DSPy example containing lead context
        prediction: Pipeline prediction with 'profile' and 'blackboard' fields
        trace: Optional trace object (for DSPy compatibility)
        pred_name: Optional predictor name (for GEPA compatibility)
        pred_trace: Optional predictor trace (for GEPA compatibility)

    Returns:
        Score between 0.0 and 1.0 (normalized from 0-100 judge score)
    """
    try:
        judge = get_judge()

        # Extract values using safe accessor
        profile = safe_get(prediction, 'profile', None)
        blackboard_data = safe_get(prediction, 'blackboard', {})

        if profile is None or profile == "None" or len(str(profile).strip()) < 10:
            print(f"DEBUG training_metric: No valid profile found in prediction")
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

        # Get judge's score (0-100)
        score = judge.judge(
            profile=str(profile),
            blackboard=blackboard_str
        )

        # Normalize to 0.0-1.0
        normalized_score = score / 100.0
        return normalized_score
        
    except Exception as e:
        print(f"ERROR in training_metric: {e}")
        return 0.0


