"""
Few-shot examples for the Profile Judge.

These examples help calibrate the judge's scoring of profile quality.
"""

from typing import List, Tuple


def get_judge_examples() -> List[Tuple[str, str, int, str]]:
    """
    Get few-shot examples for the profile judge.
    
    Returns:
        List of tuples: (blackboard, profile, score, rationale)
    """
    # TODO: Add real examples from human-scored profiles
    # For now, return empty list - judge will work without examples
    return []


def get_formatted_examples() -> str:
    """
    Format examples for the system prompt.
    
    Returns:
        Formatted string with all examples, or empty string if no examples.
    """
    examples = get_judge_examples()
    
    if not examples:
        return "(No examples available yet - using rubric only)"
    
    formatted = []
    for i, (blackboard, profile, score, rationale) in enumerate(examples, 1):
        formatted.append(f"""
### Example {i}

**Profile:**
{profile}

""")
    
    return "\n".join(formatted)

