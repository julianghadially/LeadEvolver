"""
Static LLM Judge for evaluating lead profile quality.

This judge is NOT a DSPy module - it's a static component that uses
the profile rubric to evaluate the quality of profiles produced by
the LeadProfilerPipeline.
"""

import os
import re
from typing import Optional
from openai import OpenAI
from src.context_.context import openai_key
from data.icp_context import offering, icp_profile
from src.context_.settings import judge_model

from data.profile_rubric import profile_rubric, profile_response_format
from .judge_examples import get_formatted_examples


class ProfileJudge:
    """
    Static LLM-based judge for evaluating Profile quality.
    """


    def __init__(
        self,
        model: str = judge_model
    ):
        """
        Initialize the Profile Judge.

        Args:
            model: The model to use for judging (default: see src.context_.settings.py)
        """
        self.model = model
        self.temperature = 0.0
        self.client = OpenAI(api_key=openai_key or os.getenv("OPENAI_API_KEY"))

        # Cache the formatted examples
        self._examples_prompt = get_formatted_examples()

    def _build_system_prompt(self) -> str:
        """Build the system prompt with rubric and examples."""
        return f"""You are an expert evaluator of sales lead profiles.

Your task is to score a lead profile based on the rubric provided. You are evaluating the QUALITY of the profile's composition, not how good of a sales lead they are.

## Our Ideal Customer Profile (for context on relevance)
{icp_profile}

## Offering
{offering}

## Evaluation Rubric
{profile_rubric}

## Examples of good profiles
{self._examples_prompt}

-----------------
## Your Task
Score the profile based on the rubric. Provide:
1. A breakdown score for each category
2. A total score out of 100
3. Brief rationale for each category

Respond in this exact format:
{profile_response_format}
"""

    def _build_user_prompt(
        self,
        profile: str,
        blackboard: str,
    ) -> str:
        """Build the user prompt for evaluation."""
        profile_length = len(profile) if profile else 0
        
        return f"""Evaluate the following profile:

## Research on the Lead (Blackboard):
{blackboard}

## Profile to Evaluate:
{profile}

Total Characters in profile: {profile_length}

Score this profile based on the rubric."""

    def _parse_score(self, response: str) -> int:
        """
        Parse the total score from the judge's response.
        
        Args:
            response: The raw response from the LLM
            
        Returns:
            Score between 0 and 100
        """
        # Try to find TOTAL: X/100 pattern
        total_match = re.search(r'TOTAL:\s*(\d+)\s*/\s*100', response, re.IGNORECASE)
        if total_match:
            return min(100, max(0, int(total_match.group(1))))
        
        # Fallback: try to find any number followed by /100
        fallback_match = re.search(r'(\d+)\s*/\s*100', response)
        if fallback_match:
            return min(100, max(0, int(fallback_match.group(1))))
        
        # Last resort: look for just a number at the end
        number_match = re.search(r'(\d+)\s*$', response)
        if number_match:
            score = int(number_match.group(1))
            if 0 <= score <= 100:
                return score
        
        # Default to 50 if we can't parse
        return 50

    def _parse_breakdown(self, response: str) -> dict:
        """
        Parse the category breakdown from the judge's response.
        
        Args:
            response: The raw response from the LLM
            
        Returns:
            Dict with category scores
        """
        breakdown = {}
        categories = ['ACCURACY', 'SUCCINCT', 'RELEVANT', 'COMPLETE', 'CONTACT', 'PERSONA']
        max_scores = {'ACCURACY': 60, 'SUCCINCT': 10, 'RELEVANT': 10, 'COMPLETE': 10, 'CONTACT': 10, 'PERSONA': 10}
        
        for category in categories:
            pattern = rf'{category}:\s*(\d+)\s*/\s*\d+'
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                max_score = max_scores[category]
                breakdown[category.lower()] = min(max_score, max(0, int(match.group(1))))
            else:
                breakdown[category.lower()] = 0
        
        return breakdown

    def judge(
        self,
        profile: str,
        blackboard: str,
    ) -> int:
        """
        Judge a profile's quality.

        Args:
            profile: The profile text to evaluate
            blackboard: The research context (blackboard) the profile was based on

        Returns:
            Score between 0 and 100
        """
        if not profile or len(profile.strip()) < 10:
            return 0
        
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(profile, blackboard)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=self.temperature,
            max_tokens=500
        )

        raw_response = response.choices[0].message.content.strip()
        return self._parse_score(raw_response)

    def judge_with_breakdown(
        self,
        profile: str,
        blackboard: str,
    ) -> dict:
        """
        Judge a profile's quality with detailed breakdown.

        Args:
            profile: The profile text to evaluate
            blackboard: The research context (blackboard) the profile was based on

        Returns:
            Dict with:
                - total: Total score (0-100)
                - breakdown: Dict of category scores
                - raw_response: The full judge response
        """
        if not profile or len(profile.strip()) < 10:
            return {
                "total": 0,
                "breakdown": {
                    "accuracy": 0, "succinct": 0, "relevant": 0,
                    "complete": 0, "contact": 0, "persona": 0
                },
                "raw_response": "Profile is empty or too short"
            }
        
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(profile, blackboard)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=self.temperature,
            max_tokens=500
        )

        raw_response = response.choices[0].message.content.strip()
        
        return {
            "total": self._parse_score(raw_response),
            "breakdown": self._parse_breakdown(raw_response),
            "raw_response": raw_response
        }
