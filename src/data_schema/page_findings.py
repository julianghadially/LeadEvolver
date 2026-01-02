from pydantic import BaseModel, Field
from typing import Optional

class PageFindings(BaseModel):
    """
    A class that contains the findings from a page
    """
    url: str = Field(description="the url of the page")
    title: str = Field(description="the title of the page")
    summary: str = Field(description="A summary of the page content")
    page_findings: str = Field(description="A list of findings from the page, relative to the research goal or topic area")
    interesting_links: Optional[str] = Field(description="A bar-delimited list of interesting links on the page, including the link title and the url itself.")
    current_goal: str = Field(description="The current research goal or goals")
    def to_string(self):
        return f"URL: {self.url}\nTitle: {self.title}\nSummary: {self.summary}\nPage Findings: {self.page_findings}\nCurrent Goal: {self.current_goal}\n\n"