import dspy
from data.icp_context import offering, icp_profile
from src.LeadEvolver.modules.researcher_module import ProfileResearcherModule
from src.LeadEvolver.modules.lead_profiler_module import LeadProfilerModule
from src.LeadEvolver.modules.lead_classifier_module import LeadClassifierModule
from src.data_schema.blackboard import Blackboard
import traceback
from src.tools.general_tools import load_blackboard_from_cache


class LeadProfilerPipeline(dspy.Module):
    """
    Pipeline that orchestrates profile drafting and additional research if needed.

    The pipeline:
    1. Takes initial lead info (name, URL) and any past research from the blackboard
    2. Attempts to draft a profile
    4. If more information is needed, loops back to profile researcher
    5. Continues until profile is final or max iterations reached
    """

    MAX_INVESTIGATION_ROUNDS = 3

    def __init__(self, use_system_cache: bool = True, update_classification: bool = True):
        super().__init__()
        self.researcher = ProfileResearcherModule()
        self.classifier = LeadClassifierModule()
        self.profiler = LeadProfilerModule()
        self.use_system_cache = use_system_cache
        self.update_classification = update_classification
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
        if self.use_system_cache:
            blackboard = load_blackboard_from_cache(lead_username)
        else:
            blackboard = Blackboard()
        if blackboard.to_string() == "":
            raise Exception("Missing blackboard in lead profile. There should be a blackboard to start with")
        investigation_rounds = 0

        
        # Profile drafting loop with iterative investigation
        output_ready = False
        for _ in range(self.MAX_INVESTIGATION_ROUNDS):
            # Attempt classification (classifier expects string)
            result = self.profiler(blackboard=blackboard)
            #add icp and offering to the profiler

            # If classification is final, we're done
            profile = result.profile if hasattr(result, "profile") else None
            research_goal = result.research_goal if hasattr(result, "research_goal") else None
            if research_goal is not None and len(str(research_goal)) > 5:
                #further research is needed
                try:
                    blackboard = self.researcher(
                        research_goal=research_goal,
                        blackboard=blackboard
                    )
                    investigation_rounds += 1
                except Exception as e:
                    print(f"ERROR in follow-up research: {e}")
                    traceback.print_exc()
                    # Continue with existing blackboard if research fails
                    break
            else:
                output_ready = True

        # If we've exhausted iterations, or errored out
        if not output_ready:
            result = self.profiler(blackboard=blackboard.to_string())
            profile = result.profile if hasattr(result, "profile") else None
            research_goal = result.research_goal if hasattr(result, "research_goal") else None

        if self.update_classification:
            final_classification = self.classifier(
                lead_context=blackboard.to_string(),
                force_classification=True
            )
            lead_quality = final_classification["lead_quality"] if hasattr(final_classification, "lead_quality") else None
            lead_quality_rationale = final_classification["rationale"] if hasattr(final_classification, "rationale") else None
        else:
            lead_quality = None
            lead_quality_rationale = None
        return {
            "lead_quality": lead_quality,
            "lead_quality_rationale": lead_quality_rationale,
            "profile": profile,
            "research_goal": research_goal,
            "blackboard": blackboard.to_dict(),
            "investigation_rounds": investigation_rounds
        }