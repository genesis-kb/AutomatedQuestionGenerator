"""Tavily web search over an allow-list of domains, via the REST API (no SDK dependency)."""

from __future__ import annotations

from urllib.parse import urlparse

import httpx

from aqg.logging_setup import get_logger
from aqg.schemas.grounding import Reference
from aqg.search.base import select_passages
from aqg.search.cache import JsonCache

log = get_logger("search.tavily")

TAVILY_URL = "https://api.tavily.com/search"


def host_allowed(url: str, allow_domains: list[str]) -> bool:
    """True if the URL's host equals an allowed domain or is a subdomain of one."""
    host = (urlparse(url).hostname or "").lower().rstrip(".")
    if not host:
        return False
    for domain in allow_domains:
        d = domain.lower().strip().lstrip(".").rstrip(".")
        if d and (host == d or host.endswith("." + d)):
            return True
    return False


def _kind_for(url: str, passage: str) -> str:
    if "github.com/bitcoin/bips" in url or "bips.dev" in url:
        return "spec"
    if "test vector" in passage.lower():
        return "test_vector"
    return "explainer"


class TavilySearch:
    def __init__(self, api_key: str | None, allow_domains: list[str], cache: JsonCache,
                 passage_chars: int = 800, timeout: float = 30.0,
                 transport: httpx.BaseTransport | None = None):
        self._api_key = api_key
        self.allow_domains = list(allow_domains)
        self.cache = cache
        self.passage_chars = passage_chars
        self.timeout = timeout
        self._transport = transport
        self._warned_no_key = False

    def search(self, query: str, max_results: int = 5) -> list[Reference]:
        if not self._api_key:
            if not self._warned_no_key:
                log.warning("TAVILY_API_KEY not set; web search disabled")
                self._warned_no_key = True
            return []
        if not self.allow_domains:
            log.warning("search.allow_domains is empty; web search disabled (fail closed)")
            return []

        key =self.cache.key("tavily", query, str(max_results), ",".join(sorted(self.allow_domains)))
        cached = self.cache.get(key)
        if cached is not None:
            log.debug("tavily cache hit", extra={"query": query, "results": len(cached)})
            return [Reference.model_validate(r) for r in cached]

        payload = {
            "query": query,
            "search_depth": "basic",
            "max_results": max_results,
            "include_domains": self.allow_domains,
            "include_raw_content": True,
        }
        try:
            with httpx.Client(timeout=self.timeout, transport=self._transport) as client:
                resp = client.post(TAVILY_URL, json=payload, headers={"Authorization": f"Bearer {self._api_key}"})
            log.info("tavily search", extra={"query": query, "status": resp.status_code})
            resp.raise_for_status()
            data = resp.json()
        except (httpx.HTTPError, ValueError) as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            log.warning("tavily search failed; returning no results",
                        extra={"query": query, "status": status, "error": type(exc).__name__})
            return []

        refs: list[Reference] = []
        for item in data.get("results") or []:
            url = item.get("url") or ""
            if not host_allowed(url, self.allow_domains):
                log.debug("tavily result outside allow-list dropped", extra={"url": url})
                continue
            content = item.get("content") or ""
            source = item.get("raw_content") or content
            best = select_passages(source, query, self.passage_chars, 1)
            passage = best[0] if best else content[: self.passage_chars]
            if not passage.strip():
                continue
            refs.append(Reference(url=url, title=item.get("title") or "", kind=_kind_for(url, passage),
                                  passage=passage, provider="tavily"))
        self.cache.set(key, [r.model_dump(mode="json") for r in refs])
        log.debug("tavily results", extra={"query": query, "kept": len(refs),
                                           "returned": len(data.get("results") or [])})
        return refs
