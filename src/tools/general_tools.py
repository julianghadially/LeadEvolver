import json
from pathlib import Path
from src.data_schema.blackboard import Blackboard

def clean_llm_outputted_url(url: str) -> str:
    """Clean a URL for LLM use."""
    clean_url = url.strip().strip('"').strip("'")
    
    # Handle JSON object case (e.g., {"anyOf": ["url", null]})
    if clean_url.startswith("{"):
        try:
            parsed = json.loads(clean_url)
            if isinstance(parsed, dict) and "anyOf" in parsed:
                # Extract first non-null URL from anyOf
                for item in parsed["anyOf"]:
                    if item and isinstance(item, str):
                        clean_url = item.strip('"').strip("'")
                        break
        except json.JSONDecodeError:
            pass
    return clean_url

def find_project_root() -> Path:
    """Find the project root directory by looking for marker files.
    
    Returns:
        Path to project root directory.
    """
    current = Path(__file__).resolve()
    
    # Look for project root markers
    markers = ['requirements.txt', '.git', 'readme.md']
    
    for parent in current.parents:
        # Check if any marker file exists in this directory
        if any((parent / marker).exists() for marker in markers):
            return parent

    else:
        raise ValueError("Project root not found")

def load_blackboard_from_cache(lead_username: str) -> Blackboard:
    """Load a blackboard from the cache."""
    cache_path = find_project_root() / "cache" / "system" / lead_username / "blackboard.json"
    if cache_path.exists():
        with open(cache_path, "r") as f:
            return Blackboard.from_dict(json.load(f))
    else:
        return Blackboard()