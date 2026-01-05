"""
DSPy optimization for the LeadProfiler pipeline.

Uses the training_metric (LLM judge) to optimize the pipeline
without access to ground truth labels.

This optimizer assumes that a previous process has already run to populate
the system cache with blackboard data for each lead.

Usage
```
python src/LeadEvolver/optimizer/optimize_profiler.py
```
"""

import dspy
import json
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Literal
from src.context_.settings import reflection_lm, lm_model
import mlflow
import argparse

from src.LeadEvolver.judge.profiler.metrics import training_metric
from src.LeadEvolver.data_pipeline import prepare_train_test_split
from src.LeadEvolver.modules.lead_profiler_pipeline import LeadProfilerPipeline
from src.context_.context import openai_key
from src.tools.general_tools import find_project_root


def check_cache_availability(examples: List[dspy.Example]) -> tuple[List[dspy.Example], List[str]]:
    """
    Check that cache is available for all training examples.
    
    Args:
        examples: List of dspy.Example objects with lead_username
        
    Returns:
        Tuple of (examples_with_cache, missing_usernames)
    """
    project_root = find_project_root()
    cache_dir = project_root / "cache" / "system"
    
    examples_with_cache = []
    missing_usernames = []
    
    for example in examples:
        username = example.lead_username
        cache_path = cache_dir / username / "blackboard.json"
        
        if cache_path.exists():
            examples_with_cache.append(example)
        else:
            missing_usernames.append(username)
    
    return examples_with_cache, missing_usernames


def optimize_pipeline(
    pipeline: dspy.Module,
    optimizer_type: Literal["gepa"] = "gepa",
    num_threads: int = 50,
) -> dspy.Module:
    """
    Optimize the LeadProfiler pipeline using DSPy.

    Uses the training_metric which evaluates via LLM judge
    (no ground truth access during optimization).

    Args:
        pipeline: The DSPy module to optimize
        optimizer_type: "gepa" for GEPA
        num_threads: Number of threads for parallel evaluation (default: 50)
        
    Returns:
        Optimized pipeline module
    """
    # Load training data
    trainset, _ = prepare_train_test_split()

    # Check cache availability for all training examples
    print("Checking cache availability for training examples...")
    trainset_with_cache, missing = check_cache_availability(trainset)
    
    if missing:
        print(f"\nWARNING: {len(missing)} training examples missing cache:")
        for username in missing[:10]:  # Show first 10
            print(f"  - {username}")
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more")
        print()
        
        if not trainset_with_cache:
            raise ValueError(
                "No training examples have cache available. "
                "Run the classifier pipeline first to populate cache."
            )
        
        print(f"Proceeding with {len(trainset_with_cache)}/{len(trainset)} examples that have cache.")
        trainset = trainset_with_cache
    else:
        print(f"All {len(trainset)} training examples have cache available.")

    print(f"\nOptimizing pipeline with {len(trainset)} training examples")
    print(f"Optimizer: {optimizer_type}")
    auto_setting = "light"

    if optimizer_type == "gepa":
        try:
            optimizer = dspy.GEPA(
                metric=training_metric,
                auto=auto_setting,
                reflection_lm=reflection_lm,
                num_threads=num_threads
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

    # Save results
    output_dir = "results/profiler_optimization/"
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
        "training_examples": len(trainset),
        "program_file": str(program_file.relative_to(output_path)) if program_file.exists() else None
    }
    
    results_file = output_path / f"{optimizer_type}_optimization_{timestamp}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {results_file}")

    return optimized


if __name__ == "__main__":
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Optimize LeadProfiler pipeline using DSPy")
    parser.add_argument(
        "--use-mlflow",
        action="store_true",
        default=False,
        help="Enable MLflow logging (default: False)"
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="LeadProfiler_Optimization",
        help="MLflow experiment name (default: LeadProfiler_Optimization)"
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=50,
        help="Number of threads for parallel evaluation (default: 50, max recommended for Firecrawl)"
    )
    parser.add_argument(
        "--skip-cache-check",
        action="store_true",
        default=False,
        help="Skip the cache availability check (default: False)"
    )
    args = parser.parse_args()
    
    # Configure DSPy
    lm = dspy.LM(lm_model, api_key=openai_key)
    dspy.configure(lm=lm, num_threads=args.num_threads)
    
    # Create pipeline with cache enabled
    pipeline = LeadProfilerPipeline(use_system_cache=True, update_classification=True)
    
    # Load training data for MLflow logging
    trainset, testset = prepare_train_test_split()
    
    # Start MLflow run if requested
    if args.use_mlflow:
        mlflow.dspy.autolog(log_compiles=False, log_evals=True, log_traces_from_compile=True)
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment(args.experiment_name)
        with mlflow.start_run(run_name=f"gepa_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Run optimization
            optimized = optimize_pipeline(pipeline, num_threads=args.num_threads)
            
            # Log artifacts (program and metadata are saved in optimize_pipeline)
            output_path = Path("results/profiler_optimization")
            if output_path.exists():
                # Find the most recent files to log
                program_files = sorted(output_path.glob("optimized_program_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
                results_files = sorted(output_path.glob("gepa_optimization_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
                
                if program_files:
                    mlflow.log_artifact(str(program_files[0]), "optimized_programs")
                if results_files:
                    mlflow.log_artifact(str(results_files[0]), "metadata")
            
    else:
        # Run without MLflow
        optimized = optimize_pipeline(pipeline, num_threads=args.num_threads)
    
    print("Optimized profiler pipeline ready!")

