"""
LeadEvolver - Lead Research and Classification System

A DSPy-based compound AI system for researching and classifying sales leads.
Uses iterative web research and LLM-based classification.

Usage:
    python -m src.main --mode classify --name "John Doe" --url "https://github.com/johndoe"
    python -m src.main --mode experiment1
    python -m src.main --mode demo
"""

import argparse
import dspy
from pathlib import Path

from src.context_.context import openai_key
from src.LeadEvolver.modules.lead_evolver_pipeline import LeadEvolverPipeline
from src.LeadEvolver.modules.researcher_module import ResearcherModule
from src.LeadEvolver.modules.lead_classifier_module import LeadClassifierModule

#TBD: will create main function later.