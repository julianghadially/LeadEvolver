import dspy
from src.LeadEvolver.signatures.LeadProfiler import LeadProfiler
from typing import Optional, Literal
from src.data_schema.blackboard import Blackboard
from data.icp_context import offering, icp_profile
class LeadProfilerModule(dspy.Module):
    """
    Lead profiler module drafts a profile for a lead based on the research.
    """
    def __init__(self):
        super().__init__()
        self.profiler = dspy.Predict(LeadProfiler)

    def forward(self, blackboard: Blackboard) -> dict:
        return self.profiler(blackboard=blackboard, ideal_customer_profile=icp_profile, offering=offering)