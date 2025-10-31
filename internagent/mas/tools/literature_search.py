"""
Literature Search Tool for InternAgent

This module provides tools for scientific literature search, citation management, and metadata extraction.
It integrates with multiple academic search engines and databases.
"""

import os
import asyncio
import logging
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

import aiohttp
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


@dataclass
class PaperMetadata:
    """Data class for paper metadata."""
    
    title: str
    authors: List[str]
    abstract: str
    year: Optional[int] = None
    doi: Optional[str] = None
    journal: Optional[str] = None
    url: Optional[str] = None
    citations: Optional[int] = None
    references: Optional[List[str]] = None
    keywords: Optional[List[str]] = None
    full_text: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "title": self.title,
            "authors": self.authors,
            "abstract": self.abstract,
            "year": self.year,
            "doi": self.doi,
            "journal": self.journal,
            "url": self.url,
            "citations": self.citations,
            "references": self.references,
            "keywords": self.keywords
        }
    
    def to_citation(self, format_type: str = "apa") -> str:
        """
        Generate a formatted citation.
        
        Args:
            format_type: Citation format ("apa", "mla", "chicago", "harvard", "bibtex")
            
        Returns:
            Formatted citation string
        """
        if format_type == "apa":
            # APA format
            author_text = ""
            if self.authors:
                if len(self.authors) == 1:
                    author_text = f"{self.authors[0]}."
                elif len(self.authors) == 2:
                    author_text = f"{self.authors[0]} & {self.authors[1]}."
                else:
                    author_text = f"{self.authors[0]} et al."
            
            year_text = f" ({self.year})." if self.year else ""
            journal_text = f" {self.journal}," if self.journal else ""
            doi_text = f" doi:{self.doi}" if self.doi else ""
            
            return f"{author_text}{year_text} {self.title}.{journal_text}{doi_text}"
            
        elif format_type == "bibtex":
            # BibTeX format
            first_author = self.authors[0].split(" ")[-1] if self.authors else "Unknown"
            year = self.year or "Unknown"
            key = f"{first_author}{year}"
            
            authors = " and ".join(self.authors) if self.authors else "Unknown"
            
            return (
                f"@article{{{key},\n"
                f"  author = {{{authors}}},\n"
                f"  title = {{{self.title}}},\n"
                f"  journal = {{{self.journal or 'Unknown'}}},\n"
                f"  year = {{{self.year or 'Unknown'}}},\n"
                f"  doi = {{{self.doi or ''}}}\n"
                f"}}"
            )
            
        # Default to a basic citation
        authors = ", ".join(self.authors) if self.authors else "Unknown"
        year = f"({self.year})" if self.year else ""
        journal = f"{self.journal}" if self.journal else ""
        
        return f"{authors} {year}. {self.title}. {journal}"


class CitationManager:
    """
    Manager for handling citations and bibliography.
    """
    
    def __init__(self):
        """Initialize the citation manager."""
        self.papers: Dict[str, PaperMetadata] = {}  # DOI -> PaperMetadata
        self.cached_search_results: Dict[str, List[PaperMetadata]] = {}
        
    def add_paper(self, paper: PaperMetadata) -> None:
        """
        Add a paper to the citation manager.
        
        Args:
            paper: Paper metadata to add
        """
        if paper.doi:
            self.papers[paper.doi] = paper
        else:
            # Use title as key if no DOI
            key = paper.title.lower().strip()
            existing = False
            
            # Check if we already have this paper
            for existing_paper in self.papers.values():
                if existing_paper.title.lower().strip() == key:
                    existing = True
                    break
                    
            if not existing:
                # Add with a generated key
                generated_key = f"paper_{len(self.papers)}"
                self.papers[generated_key] = paper
    
    def clear(self) -> None:
        """Clear all papers from the manager."""
        self.papers.clear()
        self.cached_search_results.clear()

class LiteratureSearch:
    """
    Tool for searching scientific literature across multiple sources.
    """
    
    def __init__(self, 
                email: str, 
                api_keys: Optional[Dict[str, str]] = None,
                citation_manager: Optional[CitationManager] = None,
                pubmed_max_retries: int = 5,
                pubmed_retry_delay: float = 5.0,
                pubmed_rate_limit_delay: float = 0.34):
        """
        Initialize the literature search tool.
        
        Args:
            email: Email for API access (required for PubMed)
            api_keys: Dictionary of API keys for different sources
            citation_manager: Citation manager to use
            pubmed_max_retries: Maximum number of retries for PubMed API calls
            pubmed_retry_delay: Initial delay between retries (seconds), will use exponential backoff
            pubmed_rate_limit_delay: Delay between requests to respect PubMed rate limit (3 requests/sec = ~0.34s)
        """
        self.email = email
        self.api_keys = api_keys or {}
        self.citation_manager = citation_manager or CitationManager()
        
        # Get User-Agent from environment variable if set
        self.user_agent = os.getenv("USER_AGENT")
        
        # Default search parameters
        self.default_max_results = 10
        self.default_sort = "relevance"  # or "date"
        
        # Cache for search results
        self._cache = {}
        
        # PubMed API retry configuration
        self.pubmed_max_retries = pubmed_max_retries
        self.pubmed_retry_delay = pubmed_retry_delay
        self.pubmed_rate_limit_delay = pubmed_rate_limit_delay
        
        # Track last PubMed request time for rate limiting
        self._last_pubmed_request = 0.0
        # Lock to ensure thread-safe rate limiting for concurrent requests
        self._pubmed_rate_limit_lock = asyncio.Lock()
    
    async def _enforce_pubmed_rate_limit(self):
        """
        Enforce PubMed API rate limit (max 3 requests per second).
        Waits if necessary to ensure we don't exceed the limit.
        
        This method is thread-safe and handles concurrent requests properly.
        """
        async with self._pubmed_rate_limit_lock:
            current_time = time.time()
            time_since_last_request = current_time - self._last_pubmed_request
            
            if time_since_last_request < self.pubmed_rate_limit_delay:
                wait_time = self.pubmed_rate_limit_delay - time_since_last_request
                await asyncio.sleep(wait_time)
            
            self._last_pubmed_request = time.time()
    
    async def _pubmed_request_with_retry(self, session: aiohttp.ClientSession, 
                                         url: str, params: Dict[str, Any], 
                                         headers: Dict[str, str],
                                         operation: str = "request") -> Optional[Dict[str, Any]]:
        """
        Make a PubMed API request with exponential backoff retry logic.
        
        Args:
            session: aiohttp ClientSession
            url: Request URL
            params: Request parameters
            headers: Request headers
            operation: Operation name for logging (e.g., "search", "fetch")
            
        Returns:
            Dictionary with 'content_type' and 'data' if successful, None if all retries exhausted.
            For JSON responses, 'data' is a dict. For XML/text responses, 'data' is a string.
        """
        retry_count = 0
        delay = self.pubmed_retry_delay
        
        while retry_count < self.pubmed_max_retries:
            try:
                # Enforce rate limit before each request
                await self._enforce_pubmed_rate_limit()
                
                async with session.get(url, params=params, headers=headers) as response:
                    # Success
                    if response.status == 200:
                        # Read the content while response is still open
                        content_type = response.content_type
                        if 'application/json' in content_type:
                            data = await response.json()
                        else:
                            data = await response.text()
                        return {'content_type': content_type, 'data': data}
                    
                    # Rate limit error (429) - retry with backoff
                    if response.status == 429:
                        retry_after = None
                        if 'Retry-After' in response.headers:
                            try:
                                retry_after = int(response.headers['Retry-After'])
                                logger.warning(f"PubMed {operation} rate limited (429). Retry-After: {retry_after}s. Retrying in {delay}s...")
                            except ValueError:
                                logger.warning(f"PubMed {operation} rate limited (429). Retrying in {delay}s...")
                        
                        if retry_after:
                            await asyncio.sleep(retry_after)
                        else:
                            await asyncio.sleep(delay)
                        
                        delay *= 2  # Exponential backoff
                        retry_count += 1
                        continue
                    
                    # Other HTTP errors - log and retry
                    logger.warning(f"PubMed {operation} error: {response.status}. Retrying in {delay}s...")
                    await asyncio.sleep(delay)
                    delay *= 2  # Exponential backoff
                    retry_count += 1
                    
            except Exception as e:
                logger.warning(f"PubMed {operation} exception: {str(e)}. Retrying in {delay}s...")
                await asyncio.sleep(delay)
                delay *= 2  # Exponential backoff
                retry_count += 1
        
        # All retries exhausted
        logger.error(f"PubMed {operation} failed after {self.pubmed_max_retries} retries")
        return None
        
    async def search_pubmed(self,
                          query: str,
                          max_results: int = 10,
                          sort: str = "relevance",
                          **kwargs) -> List[PaperMetadata]:
        """
        Search PubMed for papers matching the query.
        
        Implements rate limiting and automatic retry with exponential backoff
        to handle PubMed API limitations (max 3 requests/second).
        
        Args:
            query: Search query
            max_results: Maximum number of results
            sort: Sort order ("relevance" or "date")
            
        Returns:
            List of paper metadata
        """
        # Build the cache key
        cache_key = f"pubmed:{query}:{max_results}:{sort}"
        if cache_key in self._cache:
            logger.info(f"Using cached results for PubMed query: {query}")
            return self._cache[cache_key]
            
        logger.info(f"Searching PubMed for: {query}")
        
        # PubMed API base URLs
        base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        search_url = f"{base_url}/esearch.fcgi"
        fetch_url = f"{base_url}/efetch.fcgi"
        
        # Search parameters
        sort_param = "relevance" if sort == "relevance" else "pub+date"
        search_params = {
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "sort": sort_param,
            "retmode": "json",
            "email": self.email,
            "tool": "search_tool"
        }
        
        # Prepare headers with User-Agent if set
        headers = {}
        if self.user_agent:
            headers["User-Agent"] = self.user_agent
        
        try:
            # ClientSession with proxy support via trust_env (reads HTTP_PROXY, HTTPS_PROXY, NO_PROXY)
            async with aiohttp.ClientSession(trust_env=True) as session:
                # First, search for matching PMIDs with retry logic
                search_result = await self._pubmed_request_with_retry(
                    session, search_url, search_params, headers, "search"
                )
                
                if search_result is None:
                    logger.error(f"PubMed search failed for query: {query}")
                    return []
                
                # Extract data from result
                search_data = search_result['data'] if isinstance(search_result['data'], dict) else {}
                pmids = search_data.get("esearchresult", {}).get("idlist", [])
                
                if not pmids:
                    logger.info(f"No PubMed results found for query: {query}")
                    return []
                
                # Now fetch details for these PMIDs with retry logic
                fetch_params = {
                    "db": "pubmed",
                    "id": ",".join(pmids),
                    "retmode": "xml",
                    "email": self.email,
                    "tool": "search_tool"
                }
                
                fetch_result = await self._pubmed_request_with_retry(
                    session, fetch_url, fetch_params, headers, "fetch"
                )
                
                if fetch_result is None:
                    logger.error(f"PubMed fetch failed for query: {query}")
                    return []
                    
                xml_data = fetch_result['data']
                papers = self._parse_pubmed_xml(xml_data)
                
                # Cache the results
                self._cache[cache_key] = papers
                
                # Add papers to citation manager
                for paper in papers:
                    self.citation_manager.add_paper(paper)
                    
                logger.info(f"Successfully retrieved {len(papers)} papers from PubMed for query: {query}")
                return papers
                        
        except Exception as e:
            logger.error(f"Error search_pubmed(): {str(e)}")
            import traceback
            logger.debug(f"PubMed search traceback: {traceback.format_exc()}")
            return []
    
    async def search_arxiv(self, 
                         query: str, 
                         max_results: int = 10, 
                         sort: str = "relevance",
                         categories: Optional[List[str]] = None,
                         **kwargs) -> List[PaperMetadata]:
        """
        Search arXiv for papers matching the query.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            sort: Sort order ("relevance" or "date")
            categories: List of arXiv categories to search
            
        Returns:
            List of paper metadata
        """
        # Build the cache key
        cats_str = ",".join(categories) if categories else "all"
        cache_key = f"arxiv:{query}:{max_results}:{sort}:{cats_str}"
        if cache_key in self._cache:
            logger.info(f"Using cached results for arXiv query: {query}")
            return self._cache[cache_key]
            
        logger.debug(f"Searching arXiv for: {query}")
        
        # arXiv API URL
        search_url = "http://export.arxiv.org/api/query"
        
        # Sort parameter
        sort_param = "relevance" if sort == "relevance" else "submittedDate"
        
        # Category filter
        cat_filter = ""
        if categories:
            cat_filter = " AND (" + " OR ".join([f"cat:{cat}" for cat in categories]) + ")"
        
        # Search parameters
        search_params = {
            "search_query": f"all:{query}{cat_filter}",
            "max_results": max_results,
            "sortBy": sort_param,
            "sortOrder": "descending"
        }
        
        # Prepare headers with User-Agent if set
        headers = {}
        if self.user_agent:
            headers["User-Agent"] = self.user_agent
        
        tries = 3
        for attempt in range(tries):
            try:
                # ClientSession with proxy support via trust_env (reads HTTP_PROXY, HTTPS_PROXY, NO_PROXY)
                async with aiohttp.ClientSession(trust_env=True) as session:
                    async with session.get(search_url, params=search_params, headers=headers) as response:
                        if response.status != 200:
                            logger.error(f"arXiv search error: {response.status}")
                            if attempt < tries - 1:
                                logger.info("Retrying in 10 seconds due to error...")
                                await asyncio.sleep(10)
                            else:
                                return []
                        else:
                            xml_data = await response.text()
                            logger.info(f'arXiv REQUEST {query} success!')
                    
                        
                        papers = self._parse_arxiv_xml(xml_data)
                        
                        # Cache the results
                        self._cache[cache_key] = papers
                        
                        logger.info(f"Get {len(papers)} papers from arXiv")
                        
                        # Add papers to citation manager
                        for paper in papers:
                            self.citation_manager.add_paper(paper)
                            
                        return papers
                        
            except Exception as e:
                logger.error(f"Error searching arXiv: {e}")
                return []
    
    async def search_semantic_scholar(self,
                                    query: str,
                                    max_results: int = 10,
                                    **kwargs) -> List[PaperMetadata]:
        """
        Search Semantic Scholar for papers matching the query.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            
        Returns:
            List of paper metadata
        """
        # Check if API key is available
        api_key = os.getenv("S2_API_KEY") or self.api_keys.get("semantic_scholar")
        if not api_key:
            logger.warning("No API key for Semantic Scholar, using limited access")
            
        # Build the cache key
        cache_key = f"semantic:{query}:{max_results}"
        if cache_key in self._cache:
            logger.info(f"Using cached results for Semantic Scholar query: {query}")
            return self._cache[cache_key]
            
        logger.info(f"Searching Semantic Scholar for: {query}")
        
        # Semantic Scholar API URL
        search_url = "https://api.semanticscholar.org/graph/v1/paper/search"
        
        # Search parameters
        search_params = {
            "query": query,
            "limit": max_results,
            # "fields": "title,abstract,authors,year,journal,url,citationCount,doi",
            "fields": "title,abstract,authors.name,year,journal.name,url,citationCount,doi"
        }
        
        # Prepare headers with API key and User-Agent if set
        headers = {}
        if api_key:
            headers["x-api-key"] = api_key
        if self.user_agent:
            headers["User-Agent"] = self.user_agent
        
        tries = 3
        for attempt in range(tries):
            search_data = {}
            papers = []
            try:
                # Rate limit between requests
                await asyncio.sleep(1)
                # ClientSession with proxy support via trust_env (reads HTTP_PROXY, HTTPS_PROXY, NO_PROXY)
                async with aiohttp.ClientSession(trust_env=True) as session:
                    async with session.get(search_url, params=search_params, headers=headers) as response:
                        if response.status != 200:
                            logger.error(f"Semantic Scholar search error: {response.status}")
                            if attempt < tries - 1:
                                logger.info("Retrying in 10 seconds due to error...")
                                await asyncio.sleep(10)
                            else:
                                return []
                        else:
                            search_data = await response.json() if response.content_type == 'application/json' else {}
                            papers = []
                    
                        for paper_data in search_data.get("data", []):
                            author_list = [author.get("name", "") for author in paper_data.get("authors", [])]
                            
                            paper = PaperMetadata(
                                title=paper_data.get("title", ""),
                                authors=author_list,
                                abstract=paper_data.get("abstract", ""),
                                year=paper_data.get("year"),
                                doi=paper_data.get("doi"),
                                journal=paper_data.get("journal", {}).get("name") if paper_data.get("journal") else None,
                                url=paper_data.get("url"),
                                citations=paper_data.get("citationCount")
                            )
                            papers.append(paper)
                        
                        # Cache the results
                        self._cache[cache_key] = papers
                        
                        for paper in papers:
                            self.citation_manager.add_paper(paper)
                        
                        return papers
      
            except Exception as e:
                logger.error(f"Error searching Semantic Scholar: {str(e)}")
                return []

        
    async def multi_source_search(self, 
                               query: str, 
                               sources: List[str] = None,
                               max_results: int = 10,
                               **kwargs) -> Dict[str, List[PaperMetadata]]:
        """
        Search multiple sources simultaneously.
        
        Args:
            query: Search query
            sources: List of sources to search
            max_results: Maximum results per source
            
        Returns:
            Dictionary mapping source names to result lists
        """
        if not sources:
            sources = ["arxiv"]

        # Prepare search tasks
        tasks = []
        for source in sources:
            if source == "arxiv":
                tasks.append(self.search_arxiv(query, max_results, **kwargs))
            elif source == "pubmed":
                tasks.append(self.search_pubmed(query, max_results, **kwargs))
            elif source == "semantic_scholar":
                tasks.append(self.search_semantic_scholar(query, max_results, **kwargs))
                
        # Execute all searches in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        combined_results = {}
        for source, result in zip(sources, results):
            if isinstance(result, Exception):
                logger.error(f"Error searching {source}: {str(result)}")
                combined_results[source] = []
            else:
                combined_results[source] = result
                
        return combined_results
    
    def _parse_pubmed_xml(self, xml_data: str) -> List[PaperMetadata]:
        """
        Parse PubMed XML response to extract paper metadata.
        
        Args:
            xml_data: XML response from PubMed
            
        Returns:
            List of paper metadata
        """
        papers = []
        soup = BeautifulSoup(xml_data, "xml")
        
        for article in soup.find_all("PubmedArticle"):
            try:
                # Extract article data
                article_data = article.find("Article")
                if not article_data:
                    continue
                    
                # Title
                title = article_data.find("ArticleTitle")
                title_text = title.text if title else ""
                
                # Abstract
                abstract_elem = article_data.find("Abstract")
                abstract_text = ""
                if abstract_elem:
                    abstract_parts = abstract_elem.find_all("AbstractText")
                    if abstract_parts:
                        abstract_text = " ".join(part.text for part in abstract_parts)
                
                # Authors
                authors = []
                author_list = article_data.find("AuthorList")
                if author_list:
                    for author in author_list.find_all("Author"):
                        last_name = author.find("LastName")
                        fore_name = author.find("ForeName")
                        
                        if last_name and fore_name:
                            authors.append(f"{fore_name.text} {last_name.text}")
                        elif last_name:
                            authors.append(last_name.text)
                
                # Journal
                journal_elem = article_data.find("Journal")
                journal_name = ""
                if journal_elem:
                    journal_title = journal_elem.find("Title")
                    if journal_title:
                        journal_name = journal_title.text
                
                # Publication Date
                pub_date_elem = journal_elem.find("PubDate") if journal_elem else None
                year = None
                if pub_date_elem:
                    year_elem = pub_date_elem.find("Year")
                    if year_elem:
                        try:
                            year = int(year_elem.text)
                        except ValueError:
                            pass
                
                # DOI
                doi = None
                article_id_list = article.find("ArticleIdList")
                if article_id_list:
                    for article_id in article_id_list.find_all("ArticleId"):
                        if article_id.get("IdType") == "doi":
                            doi = article_id.text
                            break
                
                # Create paper metadata
                paper = PaperMetadata(
                    title=title_text,
                    authors=authors,
                    abstract=abstract_text,
                    year=year,
                    doi=doi,
                    journal=journal_name
                )
                papers.append(paper)
                
            except Exception as e:
                logger.error(f"Error parsing PubMed article: {str(e)}")
        
        return papers
    
    def _parse_arxiv_xml(self, xml_data: str) -> List[PaperMetadata]:
        """
        Parse arXiv XML response to extract paper metadata.
        
        Args:
            xml_data: XML response from arXiv
            
        Returns:
            List of paper metadata
        """
        papers = []
        soup = BeautifulSoup(xml_data, "xml")
        
        for entry in soup.find_all("entry"):
            try:
                # Title
                title_elem = entry.find("title")
                title_text = title_elem.text.strip() if title_elem else ""
                
                # Abstract
                summary_elem = entry.find("summary")
                abstract_text = summary_elem.text.strip() if summary_elem else ""
                
                # Authors
                authors = []
                for author in entry.find_all("author"):
                    name_elem = author.find("name")
                    if name_elem:
                        authors.append(name_elem.text.strip())
                
                # Publication year
                published_elem = entry.find("published")
                year = None
                if published_elem:
                    try:
                        pub_date = published_elem.text.strip()
                        match = re.search(r"(\d{4})", pub_date)
                        if match:
                            year = int(match.group(1))
                    except ValueError:
                        pass
                
                # DOI and URL
                doi = None
                url = None
                for link in entry.find_all("link"):
                    href = link.get("href", "")
                    if link.get("title") == "doi":
                        doi = href.replace("http://dx.doi.org/", "")
                    elif link.get("rel") == "alternate":
                        url = href
                
                # Create paper metadata
                paper = PaperMetadata(
                    title=title_text,
                    authors=authors,
                    abstract=abstract_text,
                    year=year,
                    doi=doi,
                    journal="arXiv",
                    url=url
                )
                papers.append(paper)
                
            except Exception as e:
                logger.error(f"Error parsing arXiv entry: {str(e)}")
        
        return papers
    
    def clear_cache(self) -> None:
        """Clear the search cache."""
        self._cache.clear()
