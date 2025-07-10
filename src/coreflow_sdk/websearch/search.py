import requests

from ..utils.audit import AppLogger

logger = AppLogger(__name__)


class Search:

    def __init__(self, api_key: str, base_url: str = "https://google.serper.dev"):
        self.api_key = api_key
        self.base_url = base_url

    def google_query(self, query) -> str:
        url = f"{self.base_url}/search"
        try:
            res = requests.post(
                url, headers={"X-API-KEY": self.api_key}, json={"q": query}, timeout=10
            )
            res.raise_for_status()
            items = res.json().get("organic", [])
            if not items:
                return "No search results found."
            return "\n".join([f"{item['title']}: {item['link']}" for item in items[:3]])
        except requests.RequestException as e:
            logger.error(f"Google search failed: {e}")
            return f"Search failed: {str(e)}"
