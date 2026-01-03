from pydantic import BaseModel, Field
from typing import List, Optional
from src.data_schema.page_findings import PageFindings


class Blackboard(BaseModel):
    """
    A blackboard that accumulates research findings about a lead.
    
    The blackboard stores:
    - page_findings: Accumulated findings from individual pages
    - research_findings: High-level research summaries and outcomes
    """
    
    page_findings: str = Field(
        default="",
        description="Accumulated page findings from all researched pages"
    )
    research_findings: str = Field(
        default="",
        description="Research summaries and outcomes"
    )
    
    def to_string(self) -> str:
        """
        Convert the blackboard to a string representation.
        
        Returns:
            String representation of the blackboard
        """
        parts = []
        
        if self.research_findings:
            parts.append(f"Research Summary:\n{self.research_findings}")
        
        if self.page_findings:
            parts.append(f"\nPage Findings:\n{self.page_findings}")
        
        return "\n\n".join(parts) if parts else ""
    
    def add_page_findings(self, findings: List[PageFindings]) -> None:
        """
        Process and add new page findings to the blackboard.
        
        Args:
            findings: List of PageFindings objects to add
        """
        if not findings:
            return
        
        for finding in findings:
            if isinstance(finding, PageFindings):
                self.page_findings += "\n" + finding.to_string()
            elif isinstance(finding, dict):
                # Handle dict representation
                pf = PageFindings(**finding)
                self.page_findings += "\n" + pf.to_string()
    
    def add_research_findings(self, research_findings: str) -> None:
        """
        Process and add new research findings to the blackboard.
        
        Args:
            research_findings: New research findings string to add
        """
        if not research_findings:
            return
        
        if self.research_findings:
            # Prepend new findings, keep prior ones
            self.research_findings = f"{research_findings}\n\nPrior Research:\n{self.research_findings}"
        else:
            self.research_findings = research_findings
    
    def to_dict(self) -> dict:
        """
        Convert blackboard to dictionary format.
        
        Returns:
            Dictionary with page_findings and research_findings keys
        """
        return {
            "page_findings": self.page_findings,
            "research_findings": self.research_findings
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Blackboard":
        """
        Create a Blackboard from a dictionary.
        
        Args:
            data: Dictionary with page_findings and/or research_findings keys
            
        Returns:
            Blackboard instance
        """
        return cls(
            page_findings=data.get("page_findings", ""),
            research_findings=data.get("research_findings", "")
        )

