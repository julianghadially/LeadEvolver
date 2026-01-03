from dspy import Signature, InputField, OutputField
from typing import Literal, Optional
from src.data_schema.page_findings import PageFindings

class Researcher(Signature):
    """

    Output a list of PageFindings objects that answer the research goal. Each pagefinding object contains the findings from the page and interesting links that could be explored further.

    PageFindings object:
    - url: the url of the page
    - title: the title of the page
    - summary: a summary of the page content
    - page_findings: a list of findings from the page, relative to the research goal or topic area
    - interesting_links: a bar-delimited list of interesting links on the page, including the link title and the url itself.
    - current_goal: the current research goal or goals

    Tool Usage:
    
    1. Search Tool: Use search(query="your search query")
       - Returns a list of SearchResult objects with: title, link, snippet, position
       - Examples:
         * search(query="julianghadially site:github.com")
         * search(query="Julian Ghadially site:linkedin.com")
       - Use site:xyz.com pattern to search specific domains (e.g., site:github.com, site:linkedin.com)
       - Review the search results to identify relevant pages to scrape
    
    2. Scrape Tool: Use scrape(url="https://example.com") to get full page content
       - Returns a ScrapedPage object with: url, markdown (page content), title, success, error
       - Use the 'link' field from search results as the url parameter
       - Example: scrape(url="https://github.com/user/repo")
       - Content is automatically truncated to 10,000 characters if longer
       - PDFs are currently skipped

    Workflow:
    1. Start with a search query related to your research goal
    2. Review search results (title, snippet, link) to identify promising pages
    3. Scrape relevant pages using the links from search results
    4. Analyze scraped content and create PageFindings for each page
    5. If needed, follow interesting links found on pages or perform additional searches
    6. Once sufficient information has been gathered, you should stop researching and return the page findings.
    
    Tips:
    - For GitHub profiles: Clicking into a user's repositories will show you the types of projects they have worked on, and the Read.me files can be particularly useful for figuring out what the project is about and thus, what the developer is interested in / experienced in.
    - For LinkedIn: Search for personal profiles, posts, and/or current company pages.
    - Be selective: Only scrape pages that are likely to contain relevant information to the research goal.
    - You can chain searches: Use information from one page to refine your next search query    

    Blackboard:
    - The blackboard contains past page findings. 
    - Information in the blackboard is already known and that research does not need to be duplicated, unless there are any concerns or conflicting information.
    - Your page findings will be added to the blackboard for other agents to reference.
    """
    research_goal: str = InputField(desc="A research goal or goals for gathering more information")
    blackboard: str = InputField(desc="A blackboard of information that is already known about the lead")
    page_findings: list[PageFindings] = OutputField(desc="a PageFindings object that contains the findings from each page visited")
    research_findings: str = OutputField(desc="Outcome and discussion of the research, as it pertains to the research goal(s)")
