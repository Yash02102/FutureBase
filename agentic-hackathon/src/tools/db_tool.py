import os
import requests


def query_internal_api(endpoint: str) -> str:
    base_url = os.getenv("INTERNAL_API_BASE_URL", "http://localhost:8001")
    url = f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}"
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    return response.text