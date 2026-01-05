from dspy import Signature, InputField, OutputField
from typing import Literal, Optional

class LeadProfiler(Signature):
    """LeadProfiler with iterative investigation
    
    Draft a customer profile and a contact card for a given lead, based on available information. 
    
    Try to find any relevant information as it relates to the ideal customer profile and the offering provided. If there is no releavant information, mark the profile as none.
    """

    blackboard: str = InputField(desc="Long-form research context about a sales lead")
    ideal_customer_profile: list[str] = InputField(desc="A description of the ideal customer profile")
    offering: str = InputField(desc="A description of our offering")
    profile: Optional[str] = OutputField(desc="A profile describing the lead")
    research_goal: Optional[str] = OutputField(desc="A research goal, if further investiagion is required to draft the lead profile")

    