"""
DSPy optimization for the LeadEvolver pipeline.

Uses the training_metric (LLM judge) to optimize the pipeline
without access to ground truth labels.

Usage
```
python src/LeadEvolver/optimizer/optimize.py
```
"""

import dspy
import json
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Literal
from src.context_.settings import reflection_lm, lm_model

from src.LeadEvolver.judge import training_metric
from src.LeadEvolver.data_pipeline import prepare_train_test_split


def optimize_pipeline(
    pipeline: dspy.Module,
    optimizer_type: Literal["gepa"] = "gepa",
    #num_candidates: int = 10, #controls how many candidate prompts (programs) are evaluated
) -> dspy.Module:
    """
    Optimize the LeadEvolver pipeline using DSPy.

    Uses the training_metric which evaluates via LLM judge
    (no ground truth access during optimization).

    Args:
        pipeline: The DSPy module to optimize
        optimizer_type: "gepa" for GEPA
        
    Returns:
        Optimized pipeline module
    """
    # Load training data if not provided
    trainset, _ = prepare_train_test_split()

    print(f"Optimizing pipeline with {len(trainset)} training examples")
    print(f"Optimizer: {optimizer_type}")
    auto_setting = "light"

    if optimizer_type == "gepa":
        try:
            optimizer = dspy.GEPA(
                metric=training_metric,
                auto = auto_setting,
                reflection_lm = reflection_lm
            )
        except AttributeError:
            raise ValueError(
                "GEPA optimizer not available in this DSPy version. "
                "Try: pip install --upgrade dspy-ai"
            )
        optimized = optimizer.compile(
            pipeline,
            trainset=trainset
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")

    print("Optimization complete!")

    # Save results if output_dir is provided
    output_dir = "src/optimizer/output"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    
    # Save optimized program
    program_file = output_path / f"optimized_program_{timestamp}.json"
    try:
        optimized.save(str(program_file))
        print(f"\nOptimized program saved to: {program_file}")
    except Exception as e:
        print(f"Warning: Could not save program: {e}")
    
    # Save metadata
    results = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "auto": auto_setting,
            "reflection_lm": reflection_lm
        },
        "program_file": str(program_file.relative_to(output_path)) if program_file.exists() else None
    }
    
    results_file = output_path / f"{optimizer_type}_optimization_{timestamp}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {results_file}")

    return optimized

if __name__ == "__main__":
    from src.LeadEvolver.modules.lead_evolver_pipeline import LeadEvolverPipeline
    from src.context_.context import openai_key
    import dspy
    
    # Configure DSPy
    lm = dspy.LM(lm_model, api_key=openai_key)
    dspy.configure(lm=lm)
    
    # Create and optimize pipeline
    pipeline = LeadEvolverPipeline()
    optimized = optimize_pipeline(pipeline)
    
    print("Optimized pipeline ready!")