import time
from playwright.sync_api import sync_playwright

from ..utils.audit import AppLogger

logger = AppLogger(__name__)


class Scrape:

    def __init__(self, timeout: int = 30000, max_retries: int = 2):
        """
        Initialize the Scrape client.

        Args:
            timeout: Page load timeout in milliseconds (default 30 seconds)
            max_retries: Maximum number of retry attempts (default 2)
        """
        self.timeout = timeout
        self.max_retries = max_retries

    def browse_url(self, url: str) -> str:
        """
        Browse a URL and extract main content with retries and better error handling.

        Args:
            url: URL to scrape

        Returns:
            Extracted text content or error message
        """
        for attempt in range(self.max_retries + 1):
            try:
                with sync_playwright() as p:
                    # Launch browser with better settings
                    browser = p.chromium.launch(
                        headless=True,
                        args=[
                            "--disable-blink-features=AutomationControlled",
                            "--disable-dev-shm-usage",
                            "--no-sandbox",
                        ],
                    )

                    # Create page with realistic user agent and viewport
                    page = browser.new_page(
                        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                        viewport={"width": 1920, "height": 1080},
                    )

                    # Set extra headers to appear more like a real browser
                    page.set_extra_http_headers(
                        {
                            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                            "Accept-Language": "en-US,en;q=0.5",
                            "Accept-Encoding": "gzip, deflate, br",
                            "DNT": "1",
                            "Connection": "keep-alive",
                            "Upgrade-Insecure-Requests": "1",
                        }
                    )

                    # Navigate to URL with extended timeout
                    logger.info(
                        f"Attempting to load URL: {url} (attempt {attempt + 1})"
                    )
                    page.goto(url, timeout=self.timeout, wait_until="domcontentloaded")

                    # Wait a bit for dynamic content to load
                    page.wait_for_timeout(2000)

                    # Try to get main content from common selectors
                    main_content = None
                    content_selectors = [
                        "main",
                        "article",
                        ".content",
                        "#content",
                        ".main-content",
                        ".article-content",
                        ".post-content",
                        "[role='main']",
                        ".entry-content",
                        ".weather-summary",
                        ".current-weather",
                        ".today-weather",
                        ".forecast",
                    ]

                    for selector in content_selectors:
                        try:
                            element = page.query_selector(selector)
                            if element:
                                main_content = element.text_content()
                                if main_content and len(main_content.strip()) > 50:
                                    logger.info(
                                        f"Found main content using selector: {selector}"
                                    )
                                    break
                        except Exception as selector_error:
                            logger.debug(
                                f"Selector {selector} failed: {selector_error}"
                            )
                            continue

                    # Fallback to full body if no main content found
                    if not main_content or len(main_content.strip()) < 50:
                        try:
                            main_content = page.text_content("body")
                            logger.info("Using full body content as fallback")
                        except Exception as body_error:
                            logger.warning(f"Failed to get body content: {body_error}")
                            main_content = "Could not extract content"

                    browser.close()

                    if main_content and main_content != "Could not extract content":
                        # Clean up the content - remove excessive whitespace
                        cleaned_content = " ".join(main_content.split())
                        result = cleaned_content[:2000]  # Limit for API efficiency
                        logger.info(
                            f"Successfully scraped {len(result)} characters from {url}"
                        )
                        return result
                    else:
                        logger.warning(f"No meaningful content found on {url}")
                        return "No meaningful content found on webpage"

            except Exception as e:
                error_msg = str(e)
                logger.error(
                    f"Attempt {attempt + 1} failed to browse URL {url}: {error_msg}"
                )

                if attempt < self.max_retries:
                    # Wait before retry with exponential backoff
                    wait_time = 2**attempt
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    # Final attempt failed
                    if "Timeout" in error_msg:
                        return f"Website took too long to load (timeout after {self.timeout/1000}s). This site may be slow or blocking automated access."
                    elif "net::ERR_" in error_msg:
                        return f"Network error accessing website: {error_msg}"
                    else:
                        return f"Failed to access webpage: {error_msg}"

        return "Failed to scrape URL after all retry attempts"
