import dspy
from src.LeadEvolver.modules.researcher_module import ResearcherModule
from src.LeadEvolver.modules.lead_classifier_module import LeadClassifierModule
from src.data_schema.blackboard import Blackboard
import traceback


class LeadClassifierPipeline(dspy.Module):
    """
    Main pipeline that orchestrates lead research and classification.

    The pipeline:
    1. Takes initial lead info (name, URL, any existing context)
    2. Performs initial research to gather context
    3. Attempts classification
    4. If classifier requests more investigation, loops back to researcher
    5. Continues until classification is final or max iterations reached
    """

    MAX_INVESTIGATION_ROUNDS = 5

    def __init__(self, use_system_cache: bool = False):
        super().__init__()
        self.researcher = ResearcherModule()
        self.classifier = LeadClassifierModule()
        self.use_system_cache = use_system_cache

    def forward(
        self,
        lead_url: str,
        lead_username: str = "",
        lead_name: str = ""
    ) -> dict:
        """
        Run the full lead evolution pipeline.

        Args:
            lead_name: Name of the lead (person or company)
            lead_url: Primary URL for the lead (e.g., GitHub profile)
            ideal_customer_profile: List of ICP descriptions
            offering: Description of offering

        Returns:
            dict with:
                - lead_quality: Final classification ("strong_fit", "weak_fit", "not_a_fit")
                - rationale: Rationale for the lead quality classification
                - blackboard: All accumulated research context (as dict)
                - investigation_rounds: Number of research rounds performed
        """
        # Initialize blackboard with any existing context
        blackboard = Blackboard()
        investigation_rounds = 0

        # Initial research goal
        initial_goal = f"""Find information related to whether they might be an ideal customer, by visiting only the initial url (profile page). 
Lead: {lead_username}
Name: {lead_name}
Initial Url: {lead_url}"""

        # Perform initial research
        try:
            blackboard = self.researcher(
                research_goal=initial_goal,
                blackboard=blackboard
            )
            investigation_rounds += 1
        except Exception as e:
            print(f"ERROR in initial research: {e}")
            traceback.print_exc()
            # Continue with empty blackboard if research fails

        # Classification loop with iterative investigation
        for _ in range(self.MAX_INVESTIGATION_ROUNDS):
            # Attempt classification (classifier expects string)
            classification = self.classifier(lead_context=blackboard.to_string())

            # If classification is final, we're done
            if classification["is_final"]:
                return {
                    "lead_quality": classification["lead_quality"],
                    "rationale": classification["rationale"],
                    "blackboard": blackboard.to_dict(),
                    "investigation_rounds": investigation_rounds,
                }

            # If more investigation is needed
            if classification["further_investigation"]:
                try:
                    blackboard = self.researcher(
                        research_goal=classification["further_investigation"],
                        blackboard=blackboard
                    )
                    investigation_rounds += 1
                except Exception as e:
                    print(f"ERROR in follow-up research: {e}")
                    traceback.print_exc()
                    # Continue with existing blackboard if research fails
                    break

        # If we've exhausted iterations, make a final classification
        final_classification = self.classifier(
            lead_context=blackboard.to_string(),
            force_classification=True
        )

        return {
            "lead_quality": final_classification["lead_quality"] or "None",
            "rationale": final_classification["rationale"],
            "blackboard": blackboard.to_dict(),
            "investigation_rounds": investigation_rounds,
        }

    