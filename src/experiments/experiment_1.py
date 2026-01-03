"""
Experiment #1: Evaluating lead classification with LLM judge

This experiment tests how well a ground truth LLM judge can tune a
web research agent + classification system.

Setup:
- Judge examples are defined in the CSV (judge_example=True) and loaded by the llm judge itself
- Train/test split is defined in the CSV (training_set column)
- Training examples: Compare system output to LLM-judged classifications
- Test examples: Compare system output to human labels directly

Assessment:
- Run the research + classification system on leads in github_users.csv
- Training examples use LLM judge for scoring
- Test examples are compared to human labels directly
"""

import dspy
import pandas as pd
from pathlib import Path

from src.LeadEvolver.judge import (
    LLMJudge,
    training_metric,
    test_set_metric,
    compute_classification_score
)
from src.LeadEvolver.judge.extract_judge_examples import normalize_classification


# Path to data (experiment_1.py -> experiments -> LeadEvolver -> src -> LeadEvolver root)
DATA_PATH = Path(__file__).parent.parent.parent.parent / "data" / "github_users.csv"


def load_dataset(csv_path: str = None) -> pd.DataFrame:
    """Load the github_users dataset."""
    path = csv_path or DATA_PATH
    return pd.read_csv(path)


def prepare_dataset_for_experiment(df: pd.DataFrame) -> tuple:
    """
    Prepare the dataset for Experiment #1.

    Uses CSV columns for assignment:
    - training_set: True for training, False for test
    - judge_example: True for judge few-shot examples (handled separately in judge folder)

    Args:
        df: DataFrame with lead data

    Returns:
        Tuple of (train_examples, test_examples) as dspy.Example lists
    """
    # Filter to only rows with labels
    labeled_df = df[df['icp_match'].notna() & (df['icp_match'] != '')]

    # Split based on training_set column (values: "train" or "test")
    train_df = labeled_df[labeled_df['training_set'].str.lower() == 'train']
    test_df = labeled_df[labeled_df['training_set'].str.lower() == 'test']

    # Convert to dspy.Example objects
    def to_examples(data: pd.DataFrame) -> list:
        examples = []
        for _, row in data.iterrows():
            ex = dspy.Example(
                username=row['username'],
                name=row['name'] if pd.notna(row['name']) else row['username'],
                url=row['url'],
                context=row.get('context', ''),
                icp_match=normalize_classification(row['icp_match']),
                icp_match_rationale=row.get('icp_match_rationale', '')
            ).with_inputs('username', 'name', 'url', 'context')
            examples.append(ex)
        return examples

    return to_examples(train_df), to_examples(test_df)


def run_experiment_1(lm_model: str = "openai/gpt-5-mini"):
    """
    Run Experiment #1: Evaluate LLM judge with limited human labels.

    Args:
        lm_model: The language model to use
    """
    from src.LeadEvolver.modules.lead_evolver_pipeline import LeadEvolverPipeline
    from src.context_.context import openai_key

    # Configure DSPy
    lm = dspy.LM(lm_model, api_key=openai_key)
    dspy.configure(lm=lm)

    # Load and prepare data
    df = load_dataset()
    train_examples, test_examples = prepare_dataset_for_experiment(df)

    print(f"Experiment #1 Setup:")
    print(f"  Training examples: {len(train_examples)}")
    print(f"  Test examples: {len(test_examples)}")
    print()

    # Initialize pipeline and judge
    pipeline = LeadEvolverPipeline()
    judge = LLMJudge()

    # Run evaluation on training samples
    train_scores = []

    print("Evaluating on training set (LLM judge)...")
    for i, example in enumerate(train_examples):
        try:
            result = pipeline(
                lead_name=example.name,
                lead_url=example.url,
                initial_context=example.context
            )

            predicted = result['lead_quality']

            # Use judge to evaluate
            judge_classification = judge.judge(
                lead_context=result['blackboard'],
                proposed_classification=predicted,
                proposed_rationale=result.get('rationale')
            )

            score = compute_classification_score(predicted, judge_classification)
            train_scores.append(score)

            print(f"  [{i+1}/{len(train_examples)}] {example.name}: "
                  f"Predicted={predicted}, Judge={judge_classification}, Score={score:.2f}")
        except Exception as e:
            print(f"  [{i+1}/{len(train_examples)}] {example.name}: Error - {e}")
            train_scores.append(0.0)

    # Run evaluation on test samples (direct comparison to human labels)
    print("\nEvaluating on test set (direct comparison to human labels)...")
    test_scores = []

    for i, example in enumerate(test_examples):
        try:
            result = pipeline(
                lead_name=example.name,
                lead_url=example.url,
                initial_context=example.context
            )

            predicted = result['lead_quality']
            ground_truth = example.icp_match

            score = compute_classification_score(predicted, ground_truth)
            test_scores.append(score)

            print(f"  [{i+1}/{len(test_examples)}] {example.name}: "
                  f"Predicted={predicted}, Truth={ground_truth}, Score={score:.2f}")
        except Exception as e:
            print(f"  [{i+1}/{len(test_examples)}] {example.name}: Error - {e}")
            test_scores.append(0.0)

    # Summary
    print("\n" + "="*60)
    print("EXPERIMENT #1 RESULTS")
    print("="*60)
    if train_scores:
        print(f"Training set accuracy (LLM judge): {sum(train_scores)/len(train_scores):.2%}")
    if test_scores:
        print(f"Test set accuracy (human labels): {sum(test_scores)/len(test_scores):.2%}")

    return {
        "train_accuracy": sum(train_scores)/len(train_scores) if train_scores else 0,
        "test_accuracy": sum(test_scores)/len(test_scores) if test_scores else 0,
        "train_scores": train_scores,
        "test_scores": test_scores
    }


if __name__ == "__main__":
    run_experiment_1()
