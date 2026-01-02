#from dspy import Signature, InputField, OutputField
#from typing import Literal, Optional

#class LeadProfiler(Signature):
#    """LeadProfiler with iterative investigation
#    
#    Draft a customer profile and a contact card for a given lead, based on available information. 
#    
#    Try to find any relevant information as it relates to the ideal customer profile and the offering provided. If there is no releavant information, mark the profile as none.
#    """
#
#    lead_context: str = InputField(desc="Long-form context about a sales lead")
#    ideal_customer_profiles: list[str] = InputField(desc="A list of ideal customer profiles")
#    offering: str = InputField(desc="A description of our offering")
#    contact_card: dict = OutputField(desc="Key-value pairs of any relevant contact details found, including email, website, LinkedIn, social media handles, etc.")
#    lead_profile: str = OutputField(desc="A draft of the lead's customer profile")
#
#    