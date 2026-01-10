import os
import urllib.parse

import requests


def web_search(query: str) -> str:
    enabled = os.getenv("WEB_SEARCH_ENABLED", "false").lower() == "true"
    if not enabled:
        return "Web search disabled. Set WEB_SEARCH_ENABLED=true to enable."

    url = "https://api.duckduckgo.com/?q={}&format=json&no_html=1".format(
        urllib.parse.quote(query)
    )
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    data = response.json()
    abstract = data.get("AbstractText") or "No abstract available."
    heading = data.get("Heading") or "Results"
    return f"{heading}: {abstract}"