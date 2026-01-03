import dspy
from src.LeadEvolver.signatures.Researcher import Researcher
from src.services.serper_service import SerperService, SearchResult
from src.services.firecrawl_service import FirecrawlService, ScrapedPage
from src.data_schema.blackboard import Blackboard

def search(query: str) -> str:
    """Search the web using Google. Returns a list of search results with title, link, and snippet.

    Args:
        query: The search query string. Use site:domain.com to search specific sites (e.g., "username site:github.com")

    Returns:
        A formatted string of search results with title, link, snippet, and position for each result.
    """
    service = SerperService()
    results = service.search(query=query, num_results=10)
    if not results:
        return "No search results found."

    output = []
    for r in results:
        output.append(f"Position {r.position}:\n  Title: {r.title}\n  Link: {r.link}\n  Snippet: {r.snippet}")
    return "\n\n".join(output)


def scrape(url: str) -> str:
    """Scrape a web page and return its content as markdown.

    Args:
        url: The URL of the page to scrape (e.g., "https://github.com/username/repo")

    Returns:
        The page content in markdown format, truncated to 10,000 characters if longer.
        Returns error message if scraping fails.
    """
    service = FirecrawlService()
    result = service.scrape(url=url)
    if not result.success:
        return f"Error scraping {url}: {result.error}"

    title_str = f"Title: {result.title}\n\n" if result.title else ""
    return f"{title_str}Content:\n{result.markdown}"


class ResearcherModule(dspy.Module):
    """
    Research module that uses DSPy ReAct to iteratively search and scrape web pages.
    Adds findings to the blackboard using PageFindings objects.

    Uses max 5 iterations to prevent excessive context growth.
    Each page is limited to 10k characters.
    """

    MAX_ACTIONS = 8 #incl scrape and search

    def __init__(self):
        super().__init__()
        self.researcher = dspy.ReAct(
            Researcher,
            tools=[search, scrape],
            max_iters=self.MAX_ACTIONS
        )

    def forward(self, research_goal: str, blackboard: Blackboard) -> Blackboard:
        """
        Execute research based on the goal and existing blackboard context.

        Args:
            research_goal: The research goal(s) to investigate
            blackboard: Blackboard instance with existing context/findings about the lead

        Returns:
            Blackboard object
        """
        blackboard_str = blackboard.to_string()

        # Run researcher
        result = self.researcher(research_goal=research_goal, blackboard=blackboard_str)

        # Extract findings from result (now strings, not objects)
        page_findings = result.page_findings or ""
        research_findings = result.research_findings or ""

        # Add page findings as string directly to blackboard
        if page_findings:
            if blackboard.page_findings:
                blackboard.page_findings += "\n\n---\n\n" + page_findings
            else:
                blackboard.page_findings = page_findings

        # Add research findings
        if research_findings:
            blackboard.add_research_findings(research_findings)

        return blackboard