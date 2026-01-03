from dspy import Signature, InputField, OutputField


class Researcher(Signature):
    """

    Output page findings for each page that answer the research goal. Each page contains structured findings from the page and interesting links that could be explored further.

    PageFindings structure:
    - url: the url of the page
    - title: the title of the page
    - summary: a summary of the page content
    - page_findings: a list of findings from the page, relative to the research goal or topic area
    - interesting_links: a bar-delimited list of interesting links on the page, including the link title and the url itself.
    - current_goal: the current research goal or goals

    Tool Usage:
    
    1. Search Tool: Use search(query="your search query")
       - Returns search results with: title, link, snippet, position
       - Examples:
         * search(query="julianghadially site:github.com")
         * search(query="Julian Ghadially site:linkedin.com")
       - Use site:xyz.com pattern to search specific domains (e.g., site:github.com, site:linkedin.com)
       - Review the search results to identify relevant pages to scrape
    
    2. Scrape Tool: Use scrape(url="https://example.com") to get full page content
       - Returns page content in markdown format
       - Example: scrape(url="https://github.com/user/repo")
       - Content is truncated to 10,000 characters if longer

    Workflow:
    1. Start with a search query related to your research goal
    2. Review search results to identify promising pages
    3. Scrape relevant pages using the links from search results
    4. Summarize findings for each page visited
    5. If needed, follow interesting links or perform additional searches
    6. Once sufficient information is gathered, stop researching
    
    Tips:
    - For GitHub profiles: You might want to access a user's repositories to see the types of projects they have worked on, and sometimes the Read.me files can be useful for figuring out what the project is about and thus, what the developer is interested in / experienced in.
    - For LinkedIn: Search for personal profiles, posts, and/or current company pages.
    - Be selective: Only scrape pages that are likely to contain relevant information to the research goal.
    - You can chain searches: Use information from one page to refine your next search query    

    Blackboard:
    - Contains past findings 
    - No need to duplicate research
    - Your findings will be added to the blackboard for other agents to reference
    """
    research_goal: str = InputField(desc="A research goal or goals for gathering more information")
    blackboard: str = InputField(desc="A blackboard of information that is already known about the lead")
    page_findings: str = OutputField(desc="Structured findings from each page visited. Format: For each page include URL, Title, Summary, Key Findings (bullet points), and any Interesting Links found.")
    research_findings: str = OutputField(desc="Outcome and discussion of the research, as it pertains to the research goal(s)")
