import dspy
from src.LeadEvolver.signatures.Researcher import Researcher
from src.services.serper_service import SerperService
from src.services.firecrawl_service import FirecrawlService

class ResearcherModule(dspy.Module):
    """
    Add findings to the blackboard using PageFindings objects.
    """
    def __init__(self):
        super().__init__()
        self.serper_service = SerperService()
        self.firecrawl_service = FirecrawlService()
        self.researcher = dspy.ReAct(Researcher, tools = [self.serper_service.search, self.firecrawl_service.scrape], max_iters = 5)

    def forward(self, research_goal: str, blackboard: str):
        result = self.researcher(research_goal=research_goal, blackboard=blackboard)
        blackboard = blackboard + result.page_findings.to_string()
        #TBD PICK UP HERE: 
        #   1. figure out how to do tool calls
        #   2. solidify the researcher logic. is it iterative through recursion on through tool calls? i think ill try iteratie through tool calls for now but worried about context getting too long.
        #   3. do we need to add available links to click on? not if it's using tool calls... (pick up from outline)
        #  Add available links?