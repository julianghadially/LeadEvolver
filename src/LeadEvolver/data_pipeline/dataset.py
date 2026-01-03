"""
Dataset loading and preparation for LeadEvolver.

Provides consistent dataset handling across the project:
- Load CSV data
- Split into train/test based on CSV column
- Convert to dspy.Example objects
"""

import dspy
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional

from src.LeadEvolver.judge.extract_judge_examples import normalize_classification


# Default path to data
DEFAULT_DATA_PATH = Path(__file__).parent.parent.parent.parent / "data" / "github_users.csv"


def load_dataset(csv_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load the github_users dataset.

    Args:
        csv_path: Optional path to CSV file. Uses default if not provided.

    Returns:
        DataFrame with lead data
    """
    path = csv_path or DEFAULT_DATA_PATH
    return pd.read_csv(path)


def to_dspy_examples(df: pd.DataFrame) -> List[dspy.Example]:
    """
    Convert DataFrame rows to dspy.Example objects.

    Args:
        df: DataFrame with lead data

    Returns:
        List of dspy.Example objects with inputs set
    """
    examples = []
    for _, row in df.iterrows():
        ex = dspy.Example(
            lead_url=row['url'],
            lead_username=row['username'],
            lead_name=row['name'] if pd.notna(row.get('name')) else row['username'],
            icp_match=normalize_classification(row['icp_match']),
            icp_match_rationale=row.get('icp_match_rationale', '')
        ).with_inputs('lead_url', 'lead_username', 'lead_name')
        examples.append(ex)
    return examples


def prepare_train_test_split(
    df: Optional[pd.DataFrame] = None,
    csv_path: Optional[str] = None
) -> Tuple[List[dspy.Example], List[dspy.Example]]:
    """
    Prepare train/test split based on CSV column assignments.

    Uses the 'training_set' column in the CSV:
    - "train" -> training set
    - "test" -> test set

    Args:
        df: Optional DataFrame. If not provided, loads from csv_path.
        csv_path: Optional path to CSV file.

    Returns:
        Tuple of (train_examples, test_examples) as dspy.Example lists
    """
    if df is None:
        df = load_dataset(csv_path)

    # Filter to only rows with labels
    labeled_df = df[df['icp_match'].notna() & (df['icp_match'] != '')]

    # Split based on training_set column (values: "train" or "test")
    train_df = labeled_df[labeled_df['training_set'].str.lower() == 'train']
    test_df = labeled_df[labeled_df['training_set'].str.lower() == 'test']

    return to_dspy_examples(train_df), to_dspy_examples(test_df)
