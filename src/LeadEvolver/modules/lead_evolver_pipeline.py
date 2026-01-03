import dspy
from src.LeadEvolver.modules.researcher_module import ResearcherModule
from src.LeadEvolver.modules.lead_classifier_module import LeadClassifierModule
from src.data_schema.page_findings import PageFindings
from typing import Optional, Literal


class LeadEvolverPipeline(dspy.Module):
    """
    Main pipeline that orchestrates lead research and classification.

    The pipeline:
    1. Takes initial lead info (name, URL, any existing context)
    2. Performs initial research to gather context
    3. Attempts classification
    4. If classifier requests more investigation, loops back to researcher
    5. Continues until classification is final or max iterations reached
    """

    MAX_INVESTIGATION_ROUNDS = 3

    def __init__(self):
        super().__init__()
        self.researcher = ResearcherModule()
        self.classifier = LeadClassifierModule()

    def forward(
        self,
        lead_name: str,
        lead_url: str,
        initial_context: str = "",
        ideal_customer_profile: Optional[list] = None,
        offering: Optional[str] = None
    ) -> dict:
        """
        Run the full lead evolution pipeline.

        Args:
            lead_name: Name of the lead (person or company)
            lead_url: Primary URL for the lead (e.g., GitHub profile)
            initial_context: Any existing context about the lead
            ideal_customer_profile: List of ICP descriptions
            offering: Description of offering

        Returns:
            dict with:
                - lead_quality: Final classification ("strong_fit", "weak_fit", "not_a_fit")
                - blackboard: All accumulated research context
                - page_findings: List of all PageFindings objects
                - investigation_rounds: Number of research rounds performed
                - rationale: Classification reasoning (from final context)
        """
        # Initialize blackboard with any existing context
        blackboard = initial_context or ""
        all_page_findings = []
        investigation_rounds = 0

        # Initial research goal
        initial_goal = f"Research {lead_name}. Start by exploring their profile at {lead_url}. " \
                       f"Look for: their current role, technical projects, AI/ML experience, " \
                       f"and any use of DSPy or similar frameworks."

        # Perform initial research
        research_result = self.researcher(
            research_goal=initial_goal,
            blackboard=blackboard
        )
        blackboard = research_result["updated_blackboard"]
        all_page_findings.extend(research_result["page_findings"])
        investigation_rounds += 1

        # Classification loop with iterative investigation
        for _ in range(self.MAX_INVESTIGATION_ROUNDS):
            # Attempt classification
            classification = self.classifier(
                lead_context=blackboard,
                ideal_customer_profile=ideal_customer_profile,
                offering=offering
            )

            # If classification is final, we're done
            if classification["is_final"]:
                return {
                    "lead_quality": classification["lead_quality"],
                    "blackboard": blackboard,
                    "page_findings": all_page_findings,
                    "investigation_rounds": investigation_rounds,
                    "rationale": self._extract_rationale(blackboard, classification["lead_quality"])
                }

            # If more investigation is needed
            if classification["further_investigation"]:
                research_result = self.researcher(
                    research_goal=classification["further_investigation"],
                    blackboard=blackboard
                )
                blackboard = research_result["updated_blackboard"]
                all_page_findings.extend(research_result["page_findings"])
                investigation_rounds += 1

        # If we've exhausted iterations, make a final classification
        final_classification = self.classifier(
            lead_context=blackboard + "\n\n[Note: Maximum investigation rounds reached. Make final classification with available information.]",
            ideal_customer_profile=ideal_customer_profile,
            offering=offering
        )

        return {
            "lead_quality": final_classification["lead_quality"] or "not_a_fit",
            "blackboard": blackboard,
            "page_findings": all_page_findings,
            "investigation_rounds": investigation_rounds,
            "rationale": self._extract_rationale(blackboard, final_classification["lead_quality"])
        }

    def _extract_rationale(self, blackboard: str, lead_quality: str) -> str:
        """
        Generate a brief rationale for the classification based on the blackboard.
        """
        # For now, return a summary. In a full implementation, this could use
        # another LLM call to summarize the key factors.
        if not blackboard:
            return f"Classified as {lead_quality} based on limited information."

        # Take last 500 chars of blackboard as context hint
        context_hint = blackboard[-500:] if len(blackboard) > 500 else blackboard
        return f"Classified as {lead_quality} based on research findings. Key context: {context_hint[:200]}..."
