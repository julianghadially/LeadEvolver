import dspy
from typing import Optional, Literal

from data.icp_context import offering, icp_profile
from src.LeadEvolver.signatures.LeadClassifier import LeadClassifier


class LeadClassifierModule(dspy.Module):
    """
    Lead classification module that determines if a lead is a strong fit, weak fit, or not a fit.

    Can request further investigation if more information is needed before making a classification.
    Uses the LeadClassifier signature which supports iterative investigation.
    """

    def __init__(self):
        super().__init__()
        self.classifier = dspy.Predict(LeadClassifier)

    def forward(
        self,
        lead_context: str,
        force_classification: bool = False
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
        
        result = self.classifier(
            lead_context=lead_context,
            ideal_customer_profile=icp_profile,
            offering=offering,
            force_classification=force_classification
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
