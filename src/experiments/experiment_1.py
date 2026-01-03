"""
Experiment #1: Evaluating LLM-judge-optimized lead classification

This experiment compares:
- Baseline: Unoptimized pipeline scored on test set (human labels)
- Treatment: Optimized pipeline scored on test set (human labels)

The optimization uses an LLM judge with few-shot examples but NO access
to ground truth labels for training examples.

Setup:
- Train/test split defined in CSV (training_set column)
- Judge examples defined in CSV (judge_example column)
- Optimization uses training_metric (LLM judge)
- Evaluation uses test_set_metric (human labels)
"""

import dspy
from pathlib import Path
from typing import Dict, List, Any

from src.LeadEvolver.data_pipeline import prepare_train_test_split
from src.LeadEvolver.optimizer import optimize_pipeline
from src.LeadEvolver.judge import (
    LLMJudge,
    training_metric,
    test_set_metric,
    compute_classification_score
)


def evaluate_on_test_set(
    pipeline: dspy.Module,
    test_examples: List[dspy.Example],
    label: str = "Pipeline"
) -> Dict[str, Any]:
    """
    Evaluate a pipeline on the test set using human labels.

    Args:
        pipeline: The pipeline to evaluate
        test_examples: Test examples with ground truth labels
        label: Label for logging

    Returns:
        Dict with scores and predictions
    """
    scores = []
    predictions = []

    print(f"\nEvaluating {label} on test set ({len(test_examples)} examples)...")

    for i, example in enumerate(test_examples):
        try:
            result = pipeline(
                lead_url=example.lead_url,
                lead_username=example.lead_username,
                lead_name=example.lead_name
            )

            predicted = result['lead_quality']
            rationale = result['rationale']
            ground_truth = example.icp_match

            score = compute_classification_score(predicted, ground_truth)
            scores.append(score)
            predictions.append({
                "name": example.name,
                "predicted": predicted,
                "rationale": rationale,
                "ground_truth": ground_truth,
                "score": score
            })

            print(f"  [{i+1}/{len(test_examples)}] {example.name}: "
                  f"Predicted={predicted}, Truth={ground_truth}, Score={score:.2f}")

        except Exception as e:
            print(f"  [{i+1}/{len(test_examples)}] {example.name}: Error - {e}")
            scores.append(0.0)
            predictions.append({
                "name": example.name,
                "predicted": None,
                "rationale": None,
                "ground_truth": example.icp_match,
                "score": 0.0,
                "error": str(e)
            })

    accuracy = sum(scores) / len(scores) if scores else 0.0

    return {
        "label": label,
        "accuracy": accuracy,
        "scores": scores,
        "predictions": predictions
    }


def run_experiment_1(
    lm_model: str = "openai/gpt-5-mini",
    optimizer_type: str = "mipro"
) -> Dict[str, Any]:
    """
    Run Experiment #1: Compare unoptimized vs optimized pipeline.

    Args:
        lm_model: The language model to use
        optimizer_type: "mipro" or "bootstrap"

    Returns:
        Dict with results for all conditions
    """
    from src.LeadEvolver.modules.lead_evolver_pipeline import LeadEvolverPipeline
    from src.context_.context import openai_key

    # Configure DSPy
    lm = dspy.LM(lm_model, api_key=openai_key)
    dspy.configure(lm=lm)

    # Load data
    train_examples, test_examples = prepare_train_test_split()

    print("=" * 60)
    print("EXPERIMENT #1: LLM-Judge-Optimized Lead Classification")
    print("=" * 60)
    print(f"Training examples: {len(train_examples)}")
    print(f"Test examples: {len(test_examples)}")
    print(f"Optimizer: {optimizer_type}")
    print()

    results = {}

    # =========================================================================
    # BASELINE: Unoptimized pipeline
    # =========================================================================
    print("-" * 60)
    print("BASELINE: Unoptimized Pipeline")
    print("-" * 60)

    baseline_pipeline = LeadEvolverPipeline()
    results["baseline_unoptimized"] = evaluate_on_test_set(
        baseline_pipeline,
        test_examples,
        label="Unoptimized"
    )

    # =========================================================================
    # TODO: Additional baselines with manually prompted modules
    # =========================================================================
    # Placeholder for future baselines:
    # - baseline_manual_v1: Manually crafted prompts (version 1)
    # - baseline_manual_v2: Manually crafted prompts (version 2)
    # - baseline_zero_shot: Zero-shot without any examples
    #
    # Example:
    # results["baseline_manual_v1"] = evaluate_on_test_set(
    #     ManuallyPromptedPipeline(),
    #     test_examples,
    #     label="Manual Prompts V1"
    # )

    # =========================================================================
    # TREATMENT: Optimized pipeline
    # =========================================================================
    print("-" * 60)
    print("TREATMENT: Optimizing Pipeline with LLM Judge")
    print("-" * 60)

    # Start with fresh pipeline for optimization
    pipeline_to_optimize = LeadEvolverPipeline()

    optimized_pipeline = optimize_pipeline(
        pipeline_to_optimize,
        trainset=train_examples,
        optimizer_type=optimizer_type,
        verbose=True
    )

    results["treatment_optimized"] = evaluate_on_test_set(
        optimized_pipeline,
        test_examples,
        label="Optimized"
    )

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 60)
    print("EXPERIMENT #1 RESULTS")
    print("=" * 60)

    for key, result in results.items():
        print(f"{result['label']}: {result['accuracy']:.2%}")

    # Calculate improvement
    baseline_acc = results["baseline_unoptimized"]["accuracy"]
    treatment_acc = results["treatment_optimized"]["accuracy"]
    improvement = treatment_acc - baseline_acc

    print()
    print(f"Improvement: {improvement:+.2%}")

    return results


if __name__ == "__main__":
    run_experiment_1()
