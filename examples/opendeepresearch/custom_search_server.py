"""
Custom Search Server for Search-R1

Adapts the tools.py search/browse backend (TOOLS_BASE_URL) to the
search-r1 expected output format:
    [{"document": {"contents": '"Title"\nBody text'}}]

Usage:
    Set search_backend="custom" in SEARCH_R1_CONFIGS, then configure:
        "custom": {
            "base_url": "http://t2vginfra.westus2.cloudapp.azure.com/search_tool",
            "browse": False,          # whether to fetch full page content
            "browse_max_tokens": 2048,
        }
"""

from __future__ import annotations

import asyncio
import unicodedata

import aiohttp

# ---------------------------------------------------------------------------
# Shared aiohttp session (mirrors tools.py pattern)
# ---------------------------------------------------------------------------
_shared_session: aiohttp.ClientSession | None = None
_session_lock = asyncio.Lock()


async def _get_shared_session() -> aiohttp.ClientSession:
    global _shared_session
    if _shared_session is None or _shared_session.closed:
        async with _session_lock:
            if _shared_session is None or _shared_session.closed:
                connector = aiohttp.TCPConnector(
                    limit=100,
                    limit_per_host=30,
                    ttl_dns_cache=300,
                    enable_cleanup_closed=True,
                )
                _shared_session = aiohttp.ClientSession(connector=connector)
    return _shared_session


def _is_chinese(text: str) -> bool:
    for c in text:
        try:
            if "CJK" in unicodedata.name(c):
                return True
        except ValueError:
            pass
    return False


def _get_snippet(item: dict) -> str:
    """Try multiple field names for the snippet/body content."""
    for key in ("snippet", "description", "body", "content", "text"):
        val = item.get(key)
        if val and isinstance(val, str) and val.strip():
            return val.strip()
    return "No snippet available."


async def _raw_search(
    base_url: str,
    query: str,
    max_num_results: int = 5,
    provider: str = "google",
    timeout: int = 300,
    retry_num: int = 10,
) -> list[dict]:
    """Call the TOOLS_BASE_URL/search endpoint and return raw items."""
    region = "us-en" if not _is_chinese(query) else "cn-zh"
    client_timeout = aiohttp.ClientTimeout(total=timeout)
    session = await _get_shared_session()

    for attempt in range(retry_num):
        try:
            async with session.post(
                url=f"{base_url}/search",
                json={
                    "query": str(query),
                    "max_num_results": max_num_results,
                    "region": region,
                    "provider": provider,
                },
                timeout=client_timeout,
            ) as response:
                # Accept any content-type to avoid ContentTypeError on 400/text responses
                result = await response.json(content_type=None)
                if isinstance(result, dict) and "items" in result:
                    return result["items"]
                return []
        except asyncio.CancelledError:
            raise
        except Exception:
            await asyncio.sleep(1)
            if attempt == retry_num - 1:
                return []

    return []


async def _raw_browse(
    base_url: str,
    url: str,
    max_tokens: int = 2048,
    timeout: int = 120,
    retry_num: int = 3,
) -> str:
    """Call the TOOLS_BASE_URL/browse endpoint and return page text."""
    client_timeout = aiohttp.ClientTimeout(total=timeout)
    session = await _get_shared_session()

    result = None
    for attempt in range(retry_num):
        try:
            async with session.post(
                f"{base_url}/browse",
                json={"url": str(url), "max_tokens": max_tokens},
                timeout=client_timeout,
            ) as response:
                result = await response.json(content_type=None)
                break
        except asyncio.CancelledError:
            raise
        except Exception:
            if attempt == retry_num - 1:
                return ""

    if result is None:
        return ""
    if result.get("overall_success"):
        return result.get("content", "")
    return ""
