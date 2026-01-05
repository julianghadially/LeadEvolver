from dspy import Signature, InputField, OutputField
from typing import Literal, Optional

class LeadClassifier(Signature):
    """LeadClassifier with iterative investigation
    
    Classify the quality of the given lead, relative to our ideal customer profile(s). 

    Lead quality is scored as three different classes: strong_fit, weak_fit, not_a_fit.
    
    Either produce a final verdict of lead quality (if sufficient information is provided), or instruct the
    researcher agent on a research goal or goals for gathering more information. 
    
    Note: Gathering more information carries a cost, and we want to avoid unnecessary investigation steps, especially for not_a_fit leads or leads that will end up being weak_fit.

    Context:
    Our offering is a prompt optimization service for continuously improving AI workflows and systems. We help optimize systems that have one or more outcome metrics, including ground truth and non-ground truth outcomes. Engineers that use DSPy already are particularly good fits for our offering.
    """

    lead_context: str = InputField(desc="Long-form context about a sales lead")
    ideal_customer_profile: list = InputField(desc="A description of our ideal customer profile(s)")
    offering: str = InputField(desc="A description of the offering being sold to the lead")
    force_classification: bool = InputField(desc="If true, return lead quality classifiction and do not seek additional research.")
    lead_quality: Optional[Literal["strong_fit","weak_fit","not_a_fit"]] = OutputField(desc="lead quality class")
    rationale: Optional[str] = OutputField(desc="rationale for the lead quality classification")
    further_investigation: Optional[str] = OutputField(desc="A research goal (or goals) for further investigation, if there is not enough information to make a lead quality classification")

    