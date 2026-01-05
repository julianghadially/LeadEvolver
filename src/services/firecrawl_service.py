"""Firecrawl API service for web page scraping."""
import time
import json
import hashlib
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional
from firecrawl import Firecrawl
from firecrawl.v2.types import PDFParser
from src.context_.context import firecrawl_key
from src.tools.general_tools import clean_llm_outputted_url, find_project_root

@dataclass
class ScrapedPage:
    """Result from scraping a web page."""
    url: str
    markdown: str
    title: Optional[str]
    success: bool
    error: Optional[str] = None


class FirecrawlService:
    """Service for web page scraping via Firecrawl API with local caching.

    Attributes:
        client: Firecrawl client instance.
        cache_dir: Directory for storing cached scraped pages.
        use_cache: Whether to use caching (default: True).
    """

    def __init__(self, use_cache: bool = True):
        """Initialize the Firecrawl service.

        Args:
            use_cache: Whether to enable URL caching (default: True).
            cache_dir: Directory for cache storage. Defaults to ./cache/firecrawl.
        """
        self.client = Firecrawl(api_key=firecrawl_key)
        self.use_cache = use_cache
        
        project_root = find_project_root()
        cache_dir = project_root / "cache" / "firecrawl"
        
        self.cache_dir = Path(cache_dir)
        if self.use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_key(self, url: str) -> str:
        """Generate a cache key (filename) from URL.
        
        Args:
            url: The URL to generate a key for.
            
        Returns:
            Safe filename based on URL hash.
        """
        # Normalize URL and create hash for filename
        normalized_url = url.strip().lower()
        url_hash = hashlib.md5(normalized_url.encode()).hexdigest()
        return f"{url_hash}.json"

    def _get_cache_path(self, url: str) -> Path:
        """Get the cache file path for a URL.
        
        Args:
            url: The URL to get cache path for.
            
        Returns:
            Path to cache file.
        """
        cache_key = self._get_cache_key(url)
        return self.cache_dir / cache_key

    def _load_from_cache(self, url: str) -> Optional[ScrapedPage]:
        """Load a scraped page from cache if it exists.
        
        Args:
            url: The URL to load from cache.
            
        Returns:
            ScrapedPage if found in cache, None otherwise.
        """
        if not self.use_cache:
            return None
            
        cache_path = self._get_cache_path(url)
        if cache_path.exists():
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Verify URL matches (handle URL normalization edge cases)
                    if data.get('url', '').lower() == url.lower():
                        return ScrapedPage(**data)
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                # Cache file corrupted, remove it
                cache_path.unlink(missing_ok=True)
                print(f"Warning: Corrupted cache file removed: {cache_path}")
        
        return None

    def _save_to_cache(self, page: ScrapedPage) -> None:
        """Save a scraped page to cache.
        
        Args:
            page: The ScrapedPage to cache.
        """
        if not self.use_cache:
            return
            
        cache_path = self._get_cache_path(page.url)
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(asdict(page), f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Warning: Failed to save to cache: {e}")

    def clear_cache(self) -> int:
        """Clear all cached pages.
        
        Returns:
            Number of cache files removed.
        """
        if not self.cache_dir.exists():
            return 0
        
        count = 0
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                cache_file.unlink()
                count += 1
            except Exception as e:
                print(f"Warning: Failed to remove cache file {cache_file}: {e}")
        
        return count

    def scrape(
        self,
        url: str,
        max_length: int = 10000,
        max_pdf_pages: int = 30,
        skip_pdfs: bool = True
    ) -> ScrapedPage:
        """Scrape a URL and return markdown content.
        
        Uses local cache to avoid re-scraping the same URLs.

        Args:
            url: URL to scrape.
            max_length: Maximum characters to return (truncate if longer).
            max_pdf_pages: Maximum PDF pages to scrape (not used if skip_pdfs=True).
            skip_pdfs: Whether to skip PDF files.

        Returns:
            ScrapedPage with markdown content or error information.
        """
        # Check cache first
        cached_page = self._load_from_cache(url)
        if cached_page is not None:
            print(f"Cache hit: {url} (loaded from cache)")
            return cached_page
        
        # Not in cache, scrape it
        start_time = time.time()
        try:
            url = clean_llm_outputted_url(url)
            if url.lower().endswith(".pdf") and skip_pdfs:
                page = ScrapedPage(
                    url=url,
                    markdown="PDF scraping is temporarily unavailable.",
                    title=None,
                    success=False
                )
                self._save_to_cache(page)
                return page
                
            result = self.client.scrape(url, formats=["markdown"])
            markdown = result.markdown
            
            # Truncate if needed to manage token costs
            if len(markdown) > max_length:
                markdown = markdown[:max_length] + "\n\n[Content truncated...]"
            
            scrape_time = time.time() - start_time
            print(f"URL scrape time. Url: {url}. \nTime: {scrape_time:.2f} seconds")
            
            page = ScrapedPage(
                url=url,
                markdown=markdown,
                title=result.metadata.title,
                success=True
            )
            
            # Save to cache
            self._save_to_cache(page)
            return page
            
        except Exception as e:
            page = ScrapedPage(
                url=url,
                markdown="",
                title=None,
                success=False,
                error=str(e)
            )
            # Cache errors too (to avoid retrying failed URLs)
            self._save_to_cache(page)
            return page