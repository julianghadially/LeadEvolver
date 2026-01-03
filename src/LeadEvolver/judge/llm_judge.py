"""
Static LLM Judge for evaluating lead classifications.

This judge is NOT a DSPy module - it's a static component that uses
few-shot in-context learning to evaluate the lead evolver pipeline's output.
"""

import os
from typing import Literal, Optional
from openai import OpenAI
from src.context_.context import openai_key
from data.icp_context import offering, icp_profile
from src.context_.settings import judge_model

from .judge_examples import get_formatted_examples


class LLMJudge:
    """
    Static LLM-based judge for evaluating lead classification quality.

    Uses few-shot in-context learning with human-labeled examples to
    evaluate classifications produced by the LeadEvolverPipeline.

    This is intentionally NOT a DSPy module - it's a static evaluator
    that doesn't get optimized during training.
    """

    VALID_CLASSIFICATIONS = ("strong_fit", "weak_fit", "not_a_fit")

    def __init__(
        self,
        model: str = judge_model
    ):
        """
        Initialize the LLM Judge.

        Args:
            model: The model to use for judging (default: see src.context_.settings.py)
            temperature: Temperature for generation (default: 0.0 for determinism)
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
        """
        self.model = model
        self.temperature = 0.0
        self.client = OpenAI(api_key=openai_key or os.getenv("OPENAI_API_KEY"))

        # Cache the formatted examples
        self._examples_prompt = get_formatted_examples()

    def _build_system_prompt(self) -> str:
        """Build the system prompt with few-shot examples."""
        return f"""You are an expert lead quality evaluator for a prompt optimization consulting service.

Your task is to evaluate whether a lead classification is correct based on the provided context.

## Ideal Customer Profile (ICP)
{icp_profile}

## Classification Guidelines
- **Strong fit**: Clear alignment with ICP, likely to engage with the offering.
- **Weak fit**: Some alignment with ICP - worth reaching out to, but with caveats (unclear if they will be interested, geographic limitations, unclear current involvement, one step removed from problem, etc.)
- **Not a fit**: Does not match ICP (big tech, not building AI workflows, competitor team, wrong tech stack, etc.)

## Reference Examples
The following are human-labeled examples demonstrating correct classifications:

{self._examples_prompt}

-----------------
## Your Task
Given a lead's context and the pipeline's proposed classification, determine what the CORRECT classification should be. Respond with ONLY one of: strong_fit, weak_fit, or not_a_fit."""

    def _build_user_prompt(
        self,
        lead_context: str,
        proposed_classification: str,
        proposed_rationale: Optional[str] = None
    ) -> str:
        """Build the user prompt for evaluation."""
        rationale_section = ""
        if proposed_rationale:
            rationale_section = f"\nProposed Rationale: {proposed_rationale}"

        return f"""Evaluate the following lead:

Lead Context:
{lead_context}

Pipeline's Proposed Classification: {proposed_classification}{rationale_section}

What is the CORRECT classification for this lead? Respond with ONLY: strong_fit, weak_fit, or not_a_fit."""

    def judge(
        self,
        lead_context: str,
        proposed_classification: str,
        proposed_rationale: Optional[str] = None
    ) -> str:
        """
        Judge a lead classification.

        Args:
            lead_context: The research context about the lead
            proposed_classification: The pipeline's classification (strong_fit, weak_fit, not_a_fit)
            proposed_rationale: Optional rationale from the pipeline

        Returns:
            The judge's determination: "strong_fit", "weak_fit", or "not_a_fit"
        """
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(
            lead_context,
            proposed_classification,
            proposed_rationale
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=self.temperature,
            max_tokens=20  # Only need a short response
        )

        raw_response = response.choices[0].message.content.strip().lower()

        # Parse the response - should be snake_case but handle variations
        if "strong_fit" in raw_response:
            return "strong_fit"
        elif "weak_fit" in raw_response:
            return "weak_fit"
        elif "not_a_fit" in raw_response:
            return "not_a_fit"
        else:
            # Default to not_a_fit if we can't parse
            return "not_a_fit"

    def evaluate_against_ground_truth(
        self,
        lead_context: str,
        proposed_classification: str,
        ground_truth: str,
        proposed_rationale: Optional[str] = None
    ) -> dict:
        """
        Evaluate a classification against ground truth.

        This method is used during DSPy optimization where we have
        human-labeled ground truth for the training examples.

        Args:
            lead_context: The research context about the lead
            proposed_classification: The pipeline's classification
            ground_truth: The human-labeled correct classification
            proposed_rationale: Optional rationale from the pipeline

        Returns:
            dict with:
                - judge_classification: What the judge determined
                - ground_truth: The actual correct answer
                - proposed: What the pipeline proposed
                - is_correct: Whether proposed matches ground truth
        """
        judge_classification = self.judge(
            lead_context,
            proposed_classification,
            proposed_rationale
        )

        return {
            "judge_classification": judge_classification,
            "ground_truth": ground_truth,
            "proposed": proposed_classification,
            "is_correct": proposed_classification == ground_truth
        }
