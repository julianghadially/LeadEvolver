import dspy
from src.LeadEvolver.signatures.LeadClassifier import LeadClassifier
from typing import Optional, Literal


class LeadClassifierModule(dspy.Module):
    """
    Lead classification module that determines if a lead is a strong fit, weak fit, or not a fit.

    Can request further investigation if more information is needed before making a classification.
    Uses the LeadClassifier signature which supports iterative investigation.
    """

    DEFAULT_ICP = [
        "Software engineers or teams building AI/ML systems that need optimization",
        "Companies using DSPy or similar prompt optimization frameworks",
        "Teams with compound AI systems involving multiple LLM calls",
        "Organizations seeking to improve accuracy or efficiency of AI workflows",
        "Engineers working on RAG, agents, or multi-hop reasoning systems"
    ]

    DEFAULT_OFFERING = """
    A prompt optimization service for continuously improving AI workflows and systems.
    We help optimize systems that have one or more outcome metrics, including ground truth
    and non-ground truth outcomes. Our service is particularly valuable for engineers
    already using DSPy or building compound AI systems.
    """

    def __init__(self):
        super().__init__()
        self.classifier = dspy.Predict(LeadClassifier)

    def forward(
        self,
        lead_context: str,
        ideal_customer_profile: Optional[list] = None,
        offering: Optional[str] = None
    ) -> dict:
        """
        Classify a lead based on the provided context.

        Args:
            lead_context: Long-form context about the lead (from blackboard/research)
            ideal_customer_profile: List of ICP descriptions (uses default if not provided)
            offering: Description of offering (uses default if not provided)

        Returns:
            dict with:
                - lead_quality: "strong_fit", "weak_fit", "not_a_fit", or None if more research needed
                - further_investigation: Research goal if more info needed, else None
                - is_final: True if classification is complete, False if needs more research
        """
        icp = ideal_customer_profile or self.DEFAULT_ICP
        offer = offering or self.DEFAULT_OFFERING

        result = self.classifier(
            lead_context=lead_context,
            ideal_customer_profile=icp,
            offering=offer
        )

        lead_quality = result.lead_quality
        further_investigation = result.further_investigation
        rationale = result.rationale

        # Determine if this is a final classification or needs more research
        is_final = lead_quality is not None and (
            further_investigation is None or
            further_investigation.strip() == "" or
            further_investigation.lower() == "none"
        )

        return {
            "lead_quality": lead_quality,
            "rationale": rationale,
            "further_investigation": further_investigation if not is_final else None,
            "is_final": is_final
        }
