"""
Centralized tool manager for CKA-Agent external tools.
Includes web search and web content fetcher, plus file read/write utilities.
"""

from typing import Dict, Any, Callable, Optional
import logging
import asyncio
import re
import os
import yaml
from pathlib import Path

# Optional: import for web search and web fetch
try:
    # Prefer official package name
    from duckduckgo_search import DDGS  # type: ignore

    DDGS_AVAILABLE = True
except ImportError:
    try:
        # Fallback legacy module name
        from ddgs import DDGS  # type: ignore

        DDGS_AVAILABLE = True
    except ImportError:
        DDGS = None
        DDGS_AVAILABLE = False

# Try to import Google Generative AI for summarization
try:
    from google import genai

    GENAI_AVAILABLE = True
except ImportError:
    genai = None
    GENAI_AVAILABLE = False

try:
    import httpx
    from bs4 import BeautifulSoup

    WEB_FETCH_AVAILABLE = True
except ImportError:
    httpx = None
    BeautifulSoup = None
    WEB_FETCH_AVAILABLE = False

# Try to import html2text for better markdown conversion
try:
    import html2text

    HTML2TEXT_AVAILABLE = True
except ImportError:
    html2text = None
    HTML2TEXT_AVAILABLE = False

# Try to import crawl4ai for enhanced web scraping
try:
    from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
    from crawl4ai.content_filter_strategy import PruningContentFilter
    from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator

    CRAWL4AI_AVAILABLE = True
except ImportError:
    AsyncWebCrawler = None
    BrowserConfig = None
    CrawlerRunConfig = None
    CacheMode = None
    PruningContentFilter = None
    DefaultMarkdownGenerator = None
    CRAWL4AI_AVAILABLE = False

logger = logging.getLogger("CKA-Tools")


# -------------------
# Helper Functions
# -------------------
def _count_words(text: str) -> int:
    """Count words in text."""
    if not text:
        return 0
    return len(text.split())


# -------------------
# Config Loading
# -------------------
def _load_config() -> Dict[str, Any]:
    """Load configuration from config.yml file."""
    try:
        config_path = (
            Path(__file__).parent.parent.parent.parent / "config" / "config.yml"
        )
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            return config
        else:
            logger.warning(f"Config file not found at {config_path}")
            return {}
    except Exception as e:
        logger.warning(f"Failed to load config: {e}")
        return {}


# Global config cache
_config_cache = None


def _get_config() -> Dict[str, Any]:
    """Get cached config or load it."""
    global _config_cache
    if _config_cache is None:
        _config_cache = _load_config()
    return _config_cache


class ToolRegistry:
    """Lightweight tool registry that maps tool-name -> callable."""

    def __init__(self):
        self._tools: Dict[str, Callable[[Dict[str, Any]], str]] = {}
        self.logger = logging.getLogger(self.__class__.__name__)

    def register(self, name: str, func: Callable[[Dict[str, Any]], str]) -> None:
        self._tools[name] = func

    def get(self, name: str) -> Optional[Callable[[Dict[str, Any]], str]]:
        return self._tools.get(name)

    def call(self, name: str, payload: Dict[str, Any]) -> str:
        func = self.get(name)
        if not func:
            return f"[tool:{name}] not found"
        try:
            return func(payload)
        except Exception as e:
            self.logger.warning(f"tool {name} failed: {e}")
            return f"[tool:{name}] error: {e}"


# -------------------
# Web Search and Fetch Tool
# -------------------
def tool_web_search_fetch(args: Dict[str, Any]) -> str:
    print(
        "tool_web_search_fetch calledxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
    )
    """Search the web using DuckDuckGo and fetch/summarize content from the top results."""
    try:
        print("Using Web Search with Content Fetching")
    except Exception:
        pass
    logger.info("Using Web Search with Content Fetching")

    query = args.get("query", "").strip()
    num_results = args.get(
        "num_results", 3
    )  # Reduced default since we'll fetch content
    fetch_content = args.get(
        "fetch_content", True
    )  # New option to disable content fetching
    safesearch = args.get("safesearch", "moderate")
    timelimit = args.get("timelimit")

    if not query:
        return "Error: No search query provided"

    if not DDGS_AVAILABLE:
        return "Error: DuckDuckGo search not available. Install: pip install duckduckgo-search"

    try:
        logger.info(f"Searching for: {query} (max {num_results} results)")

        with DDGS() as ddgs:
            search_results = list(
                ddgs.text(
                    query,
                    safesearch=safesearch,
                    timelimit=timelimit,
                    max_results=num_results,
                )
            )

        if not search_results:
            return f"No search results found for '{query}'"

        # If content fetching is disabled, return basic search results
        if not fetch_content:
            formatted_results = []
            for i, result in enumerate(search_results, 1):
                title = result.get("title", "No title")
                body = result.get("body", "No description")
                href = result.get("href", "No URL")

                formatted_result = f"{i}. {title}\n" f"   URL: {href}\n" f"   {body}"
                formatted_results.append(formatted_result)

            result_text = f"Web search results for '{query}':\n\n" + "\n\n".join(
                formatted_results
            )
            logger.info(f"Found {len(search_results)} results")
            return result_text

        # Fetch and summarize content from search results (in parallel)
        logger.info(
            f"Starting parallel content fetching from {len(search_results)} URLs..."
        )

        # Use asyncio to fetch content from all URLs in parallel
        try:
            combined_content = asyncio.run(
                _fetch_all_content_parallel(search_results, query)
            )
            successful_fetches = sum(
                1
                for content in combined_content
                if not content.startswith("**Result")
                or "Note: Could not fetch full content" not in content
            )
        except Exception as e:
            logger.error(f"Parallel fetching failed: {e}, falling back to sequential")
            # Fallback to sequential processing
            combined_content = []
            successful_fetches = 0

            for i, result in enumerate(search_results, 1):
                title = result.get("title", "No title")
                href = result.get("href", "No URL")
                body = result.get("body", "No description")

                logger.info(f"Fetching content from result {i}: {href}")

                try:
                    # Use the existing fetch_content function
                    content_args = {"url": href, "query": query}
                    fetched_content = tool_fetch_content(content_args)

                    # Check if fetch was successful (doesn't start with "Error:")
                    if not fetched_content.startswith("Error:"):
                        word_count = _count_words(fetched_content)
                        logger.info(
                            f"Successfully fetched and summarized content from {href}: {word_count} words"
                        )
                        combined_content.append(
                            f"**Result {i}: {title}**\nURL: {href}\n\n{fetched_content}"
                        )
                        successful_fetches += 1
                    else:
                        # Fallback to search snippet if fetch failed
                        combined_content.append(
                            f"**Result {i}: {title}**\nURL: {href}\nContent: {body}\n(Note: Could not fetch full content - {fetched_content})"
                        )

                except Exception as e:
                    logger.warning(f"Failed to fetch content from {href}: {e}")
                    # Fallback to search snippet
                    combined_content.append(
                        f"**Result {i}: {title}**\nURL: {href}\nContent: {body}\n(Note: Could not fetch full content)"
                    )

        result_text = (
            f"Web search and content summary for '{query}':\n\n"
            + "\n\n---\n\n".join(combined_content)
        )

        logger.info(
            f"Found {len(search_results)} results, successfully fetched content from {successful_fetches}"
        )
        return result_text

    except Exception as e:
        error_msg = f"Error: Web search failed - {str(e)}"
        logger.error(error_msg)
        return error_msg


# -------------------
# Parallel Content Fetching Helper
# -------------------
async def _fetch_all_content_parallel(search_results: list, query: str) -> list:
    """Fetch content from all search results in parallel."""

    async def fetch_single_result(result: dict, index: int) -> str:
        """Fetch content from a single search result."""
        title = result.get("title", "No title")
        href = result.get("href", "No URL")
        body = result.get("body", "No description")

        logger.info(f"Fetching content from result {index}: {href}")

        try:
            # Use async version of fetch_content
            fetched_content = await _fetch_content_async(href, query)

            if not fetched_content.startswith("Error:"):
                word_count = _count_words(fetched_content)
                logger.info(
                    f"Successfully fetched and summarized content from {href}: {word_count} words"
                )
                return f"**Result {index}: {title}**\nURL: {href}\n\n{fetched_content}"
            else:
                # Fallback to search snippet if fetch failed
                return f"**Result {index}: {title}**\nURL: {href}\nContent: {body}\n(Note: Could not fetch full content - {fetched_content})"

        except Exception as e:
            logger.warning(f"Failed to fetch content from {href}: {e}")
            # Fallback to search snippet
            return f"**Result {index}: {title}**\nURL: {href}\nContent: {body}\n(Note: Could not fetch full content)"

    # Create tasks for parallel execution
    tasks = [
        fetch_single_result(result, i + 1) for i, result in enumerate(search_results)
    ]

    # Run all tasks in parallel
    return await asyncio.gather(*tasks, return_exceptions=True)


async def _fetch_content_async(url: str, query: Optional[str] = None) -> str:
    """Async version of content fetching."""
    if not url:
        return "Error: No URL provided"

    # Try Crawl4AI first (preferred method)
    if CRAWL4AI_AVAILABLE:
        try:
            logger.info(f"Using Crawl4AI to fetch: {url}")
            return await _fetch_with_crawl4ai_async(url, query)
        except Exception as e:
            logger.warning(f"Crawl4AI failed: {e}, falling back to httpx+html2text")

    # Fallback to httpx + html2text
    if not WEB_FETCH_AVAILABLE:
        return (
            "Error: Web fetch not available. Install: pip install httpx beautifulsoup4"
        )

    try:
        logger.info(f"Fetching content from: {url} (using httpx+html2text)")

        async with httpx.AsyncClient(follow_redirects=True) as client:
            response = await client.get(url, timeout=30.0)
            response.raise_for_status()

        # Convert to markdown if html2text is available
        if HTML2TEXT_AVAILABLE:
            text = _html_to_markdown(response.text)
        else:
            # Parse HTML with BeautifulSoup as fallback
            soup = BeautifulSoup(response.text, "html.parser")

            # Remove unwanted elements
            for element in soup(
                [
                    "script",
                    "style",
                    "nav",
                    "header",
                    "footer",
                    "aside",
                    "form",
                    "iframe",
                    "noscript",
                ]
            ):
                element.decompose()

            # Extract text
            text = soup.get_text(separator=" ", strip=True)

            # Clean up whitespace
            text = re.sub(r"\s+", " ", text).strip()

        # Process content (summarize)
        text = _process_fetched_content(text, query)

        word_count = _count_words(text)
        logger.info(f"Successfully fetched {len(text)} characters ({word_count} words)")
        return text

    except httpx.HTTPStatusError as e:
        error_msg = f"Error: HTTP {e.response.status_code} - {e.response.reason_phrase}"
        logger.error(error_msg)
        return error_msg
    except httpx.TimeoutException:
        error_msg = f"Error: Request timeout after 30s"
        logger.error(error_msg)
        return error_msg
    except Exception as e:
        error_msg = f"Error: Failed to fetch content - {str(e)}"
        logger.error(error_msg)
        return error_msg


async def _fetch_with_crawl4ai_async(url: str, query: Optional[str] = None) -> str:
    """Async version of Crawl4AI fetching."""
    try:
        # Configure browser (headless mode)
        browser_config = BrowserConfig(
            headless=True,
            verbose=False,
        )

        # Configure crawler with PruningContentFilter for better content extraction
        run_config = CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,  # Always fetch fresh content
            markdown_generator=DefaultMarkdownGenerator(
                content_filter=PruningContentFilter(
                    threshold=0.48,  # Default threshold for content relevance
                    threshold_type="fixed",
                    min_word_threshold=0,
                )
            ),
        )

        async with AsyncWebCrawler(config=browser_config) as crawler:
            result = await crawler.arun(url=url, config=run_config)

            if not result.success:
                raise ValueError(f"Crawl failed: {result.error_message}")

            # Get the fit markdown (cleaned and filtered content)
            if hasattr(result.markdown, "fit_markdown"):
                content = result.markdown.fit_markdown
                if not content or len(content.strip()) < 100:
                    logger.debug("fit_markdown too short, using raw_markdown")
                    content = result.markdown.raw_markdown
            elif hasattr(result, "markdown"):
                content = (
                    result.markdown
                    if isinstance(result.markdown, str)
                    else result.markdown.raw_markdown
                )
            else:
                content = result.cleaned_html or result.html

            # Remove excessive whitespace
            content = re.sub(r"\n{3,}", "\n\n", content).strip()

            # Process content (summarize)
            content = _process_fetched_content(content, query)

            word_count = _count_words(content)
            logger.info(
                f"Crawl4AI successfully fetched {len(content)} characters ({word_count} words)"
            )
            return content

    except Exception as e:
        raise ValueError(f"Crawl4AI execution failed: {e}")


# -------------------
# Web Content Fetcher Tool with Crawl4AI
# -------------------
def tool_fetch_content(args: Dict[str, Any]) -> str:
    """Fetch and extract text content from a URL.

    By default uses Crawl4AI for best results. Falls back to httpx+html2text if needed.
    Automatically converts to markdown and summarizes content (~200 words) for better LLM consumption.
    """
    url = args.get("url", "").strip()
    timeout = args.get("timeout", 30.0)
    query = args.get("query")  # Optional query for context-aware summarization

    if not url:
        return "Error: No URL provided"

    # Try Crawl4AI first (preferred method)
    if CRAWL4AI_AVAILABLE:
        try:
            logger.info(f"Using Crawl4AI to fetch: {url}")
            return _fetch_with_crawl4ai(url, query)
        except Exception as e:
            logger.warning(f"Crawl4AI failed: {e}, falling back to httpx+html2text")

    # Fallback to httpx + html2text
    if not WEB_FETCH_AVAILABLE:
        return (
            "Error: Web fetch not available. Install: pip install httpx beautifulsoup4"
        )

    async def fetch():
        try:
            logger.info(f"Fetching content from: {url} (using httpx+html2text)")

            async with httpx.AsyncClient(follow_redirects=True) as client:
                response = await client.get(url, timeout=timeout)
                response.raise_for_status()

            # Convert to markdown if html2text is available
            if HTML2TEXT_AVAILABLE:
                text = _html_to_markdown(response.text)
            else:
                # Parse HTML with BeautifulSoup as fallback
                soup = BeautifulSoup(response.text, "html.parser")

                # Remove unwanted elements
                for element in soup(
                    [
                        "script",
                        "style",
                        "nav",
                        "header",
                        "footer",
                        "aside",
                        "form",
                        "iframe",
                        "noscript",
                    ]
                ):
                    element.decompose()

                # Extract text
                text = soup.get_text(separator=" ", strip=True)

                # Clean up whitespace
                text = re.sub(r"\s+", " ", text).strip()

            # Process content (summarize)
            text = _process_fetched_content(text, query)

            word_count = _count_words(text)
            logger.info(
                f"Successfully fetched {len(text)} characters ({word_count} words)"
            )
            return text

        except httpx.HTTPStatusError as e:
            error_msg = (
                f"Error: HTTP {e.response.status_code} - {e.response.reason_phrase}"
            )
            logger.error(error_msg)
            return error_msg
        except httpx.TimeoutException:
            error_msg = f"Error: Request timeout after {timeout}s"
            logger.error(error_msg)
            return error_msg
        except Exception as e:
            error_msg = f"Error: Failed to fetch content - {str(e)}"
            logger.error(error_msg)
            return error_msg

    try:
        return asyncio.run(fetch())
    except Exception as e:
        error_msg = f"Error: Async execution failed - {str(e)}"
        logger.error(error_msg)
        return error_msg


def _fetch_with_crawl4ai(
    url: str,
    query: Optional[str] = None,
) -> str:
    """Fetch content using Crawl4AI with PruningContentFilter.

    Automatically converts to markdown and summarizes content.
    """

    async def crawl():
        try:
            # Configure browser (headless mode)
            browser_config = BrowserConfig(
                headless=True,
                verbose=False,
            )

            # Configure crawler with PruningContentFilter for better content extraction
            # Using fit_markdown which filters out less relevant content
            run_config = CrawlerRunConfig(
                cache_mode=CacheMode.BYPASS,  # Always fetch fresh content
                markdown_generator=DefaultMarkdownGenerator(
                    content_filter=PruningContentFilter(
                        threshold=0.48,  # Default threshold for content relevance
                        threshold_type="fixed",
                        min_word_threshold=0,
                    )
                ),
            )

            async with AsyncWebCrawler(config=browser_config) as crawler:
                result = await crawler.arun(url=url, config=run_config)

                if not result.success:
                    raise ValueError(f"Crawl failed: {result.error_message}")

                # Get the fit markdown (cleaned and filtered content)
                if hasattr(result.markdown, "fit_markdown"):
                    # Use fit_markdown for cleaner, more relevant content
                    content = result.markdown.fit_markdown
                    if not content or len(content.strip()) < 100:
                        # Fallback to raw_markdown if fit_markdown is too short
                        logger.debug("fit_markdown too short, using raw_markdown")
                        content = result.markdown.raw_markdown
                elif hasattr(result, "markdown"):
                    # Fallback for older API
                    content = (
                        result.markdown
                        if isinstance(result.markdown, str)
                        else result.markdown.raw_markdown
                    )
                else:
                    # Fallback to HTML if markdown not available
                    content = result.cleaned_html or result.html

                # Remove excessive whitespace
                content = re.sub(r"\n{3,}", "\n\n", content).strip()

                return content

        except Exception as e:
            raise ValueError(f"Crawl4AI error: {e}")

    try:
        content = asyncio.run(crawl())

        # Process content (summarize)
        content = _process_fetched_content(content, query)

        word_count = _count_words(content)
        logger.info(
            f"Crawl4AI successfully fetched {len(content)} characters ({word_count} words)"
        )
        return content
    except Exception as e:
        raise ValueError(f"Crawl4AI execution failed: {e}")


def _html_to_markdown(html_content: str) -> str:
    """Convert HTML to clean markdown using html2text."""
    h = html2text.HTML2Text()
    h.ignore_links = False
    h.ignore_images = False
    h.ignore_emphasis = False
    h.body_width = 0  # Don't wrap lines
    h.skip_internal_links = True
    h.ignore_tables = False

    markdown = h.handle(html_content)

    # Clean up excessive newlines
    markdown = re.sub(r"\n{3,}", "\n\n", markdown)

    return markdown.strip()


def _process_fetched_content(
    content: str,
    query: Optional[str] = None,
) -> str:
    """Process fetched content: summarize to ~200 words.

    Args:
        content: The content to process
        query: Optional query/context for better summarization

    Returns:
        Summarized content (~200 words)
    """
    # Always summarize if content is long enough
    if len(content) > 1000:
        content = _summarize_content(content, query)

    return content


def _summarize_content(content: str, query: Optional[str] = None) -> str:
    """Summarize content using Gemini-2.5-flash model.

    Args:
        content: The content to summarize
        query: Optional query/context to help guide the summarization

    Returns:
        Summarized content (~200 words) or original content if summarization fails
    """
    if not GENAI_AVAILABLE:
        logger.warning(
            "Google Generative AI not available. Install: pip install google-generativeai"
        )
        return content

    try:
        # Load config to get API key
        config = _get_config()
        if not config or "model" not in config or "blackbox" not in config["model"]:
            logger.warning("Config not found or missing model.blackbox section")
            return content

        blackbox_config = config["model"]["blackbox"]

        # Determine which API key to use
        if blackbox_config.get("use_proxy", False):
            # For proxy mode, we can't use genai library directly
            logger.info("Proxy mode not supported for summarization, skipping")
            return content
        else:
            api_key = blackbox_config.get("api_key")
            if not api_key:
                logger.warning("No Gemini API key found in config")
                return content

        # Configure Gemini client
        client = genai.Client(api_key=api_key)

        # Create summarization prompt with optional query context
        query_context = (
            f"\n\nOriginal search query/context: {query}\nPlease focus the summary on information relevant to this query."
            if query
            else ""
        )

        prompt = f"""Please summarize the following web content into approximately 200 words.
Focus on the key points, main ideas, and important information.
Be concise and informative. These information are given based the query: "{query_context}". Please stick to the content and avoid adding your own opinions.

Content to summarize:
{content}

Summary (approximately 200 words):"""

        original_word_count = _count_words(content)
        logger.info(
            f"Requesting summarization from Gemini-2.5-flash{' with query context' if query else ''} (original: {original_word_count} words)..."
        )

        # Generate summary using the new API
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )

        if response and response.text:
            summary = response.text.strip()
            summary_word_count = _count_words(summary)
            logger.info(
                f"Successfully summarized {len(content)} chars ({original_word_count} words) to {len(summary)} chars ({summary_word_count} words)"
            )
            return summary
        else:
            logger.warning("No response from Gemini, returning original content")
            return content

    except Exception as e:
        logger.warning(f"Summarization failed: {e}, returning original content")
        return content


# -------------------
# File Read/Write Tools
# -------------------
def tool_read_file(args: Dict[str, Any]) -> str:
    """Read content from a file."""
    path = args.get("path", "").strip()
    max_chars = args.get("max_chars", 50000)

    if not path:
        return "Error: No file path provided"

    try:
        logger.info(f"Reading file: {path}")

        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        original_length = len(content)

        if max_chars > 0 and len(content) > max_chars:
            content = (
                content[:max_chars]
                + f"\n\n... [content truncated from {original_length} to {max_chars} chars]"
            )
            logger.info(f"File content truncated")

        logger.info(f"Successfully read {original_length} characters")
        return content

    except FileNotFoundError:
        error_msg = f"Error: File not found - {path}"
        logger.error(error_msg)
        return error_msg
    except PermissionError:
        error_msg = f"Error: Permission denied - {path}"
        logger.error(error_msg)
        return error_msg
    except Exception as e:
        error_msg = f"Error: Failed to read file - {str(e)}"
        logger.error(error_msg)
        return error_msg


def tool_read_files(args: Dict[str, Any]) -> str:
    """Read content from multiple files."""
    paths = args.get("paths", [])
    max_chars = args.get("max_chars", 50000)

    if not paths or not isinstance(paths, list):
        return "Error: No file paths provided (expected list)"

    if len(paths) == 0:
        return "Error: Empty file paths list"

    logger.info(f"Reading {len(paths)} files")

    results = []
    success_count = 0

    for path in paths:
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()

            if max_chars > 0 and len(content) > max_chars:
                content = (
                    content[:max_chars] + f"\n... [truncated from {len(content)} chars]"
                )

            results.append(f"--- {path} ---\n{content}\n")
            success_count += 1

        except Exception as e:
            error_msg = f"--- {path} ---\nError: {str(e)}\n"
            results.append(error_msg)
            logger.error(f"Failed to read {path}: {e}")

    logger.info(f"Successfully read {success_count}/{len(paths)} files")
    return "\n".join(results)


def tool_write_file(args: Dict[str, Any]) -> str:
    """Write content to a file."""
    path = args.get("path", "").strip()
    content = args.get("content", "")
    append = args.get("append", False)

    if not path:
        return "Error: No file path provided"

    try:
        mode = "a" if append else "w"
        logger.info(f"Writing to file: {path} (mode: {mode})")

        with open(path, mode, encoding="utf-8") as f:
            f.write(content)

        action = "Appended to" if append else "Wrote to"
        success_msg = f"Successfully {action.lower()} {path} ({len(content)} chars)"
        logger.info(success_msg)
        return success_msg

    except PermissionError:
        error_msg = f"Error: Permission denied - {path}"
        logger.error(error_msg)
        return error_msg
    except Exception as e:
        error_msg = f"Error: Failed to write file - {str(e)}"
        logger.error(error_msg)
        return error_msg


# -------------------
# Tool Registry
# -------------------
tool_registry = {
    "web_search_fetch": tool_web_search_fetch,
    "read_file": tool_read_file,
    "read_files": tool_read_files,
    "write_file": tool_write_file,
}


# -------------------
# Tool Definitions (for LLM)
# -------------------
def get_tool_definitions() -> list:
    """Get OpenAI-compatible tool definitions."""
    return [
        {
            "type": "function",
            "function": {
                "name": "web_search_fetch",
                "description": "Search the web using DuckDuckGo and automatically fetch and summarize content from the top search results. Returns comprehensive information from multiple sources.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The search query"},
                        "num_results": {
                            "type": "integer",
                            "description": "Number of results to return (1-5, default 3). Lower numbers recommended since content is fetched.",
                            "default": 3,
                        },
                        "fetch_content": {
                            "type": "boolean",
                            "description": "Whether to fetch and summarize content from search results. If false, returns only search snippets.",
                            "default": True,
                        },
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read content from a text file.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File path to read"},
                        "max_chars": {
                            "type": "integer",
                            "description": "Maximum characters to return",
                            "default": 50000,
                        },
                    },
                    "required": ["path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "write_file",
                "description": "Write content to a text file.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File path to write"},
                        "content": {
                            "type": "string",
                            "description": "Content to write",
                        },
                        "append": {
                            "type": "boolean",
                            "description": "Append instead of overwrite",
                            "default": False,
                        },
                    },
                    "required": ["path", "content"],
                },
            },
        },
    ]


# -------------------
# Utility Functions
# -------------------


def get_status_report() -> str:
    """Get a status report of available tools."""
    report = ["=" * 70, "Tool Status Report", "=" * 70, ""]

    # Show registered tools
    report.append("Registered Tools:")
    report.append("-" * 70)
    for tool_name in sorted(tool_registry.keys()):
        report.append(f"  ✓ {tool_name}")

    report.append("")
    report.append("=" * 70)
    report.append("Dependencies Status:")
    report.append("=" * 70)

    return "\n".join(report)
