"""Generic HTTP fetcher with text extraction."""

from __future__ import annotations

import httpx
import trafilatura


def url_fetch(
    url: str,
    extract_text: bool = True,
    max_chars: int = 20000,
) -> str:
    """Fetch a URL and optionally extract clean text.

    Args:
        url: The URL to fetch.
        extract_text: If True, use trafilatura for clean text extraction.
            If False, return raw HTML.
        max_chars: Maximum characters to return (truncated if longer).

    Returns:
        The fetched content as a string, truncated to max_chars.
    """
    try:
        with httpx.Client(timeout=30, follow_redirects=True) as client:
            response = client.get(url)
            response.raise_for_status()
            html = response.text
    except httpx.HTTPError as e:
        return f"Error fetching {url}: {e}"

    if extract_text:
        extracted = trafilatura.extract(html)
        content = extracted if extracted else html
    else:
        content = html

    return content[:max_chars]
