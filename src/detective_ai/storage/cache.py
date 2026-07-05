"""Simple in-memory cache for expensive computations.

Replaces Redis with a lightweight dict-based cache for local development.
"""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


class InMemoryCache:
    """Thread-safe in-memory cache with optional TTL expiry."""

    def __init__(self, default_ttl: int = 3600) -> None:
        self._store: dict[str, tuple[Any, float]] = {}  # key → (value, expiry_time)
        self._default_ttl = default_ttl

    def get(self, key: str) -> Any | None:
        """Retrieve a cached value, returning None if expired or missing."""
        if key not in self._store:
            return None
        value, expiry = self._store[key]
        if time.time() > expiry:
            del self._store[key]
            return None
        return value

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Store a value with optional TTL (seconds)."""
        expiry = time.time() + (ttl or self._default_ttl)
        self._store[key] = (value, expiry)

    def delete(self, key: str) -> bool:
        """Delete a key from the cache."""
        if key in self._store:
            del self._store[key]
            return True
        return False

    def clear(self) -> None:
        """Clear all cached values."""
        self._store.clear()

    def cleanup(self) -> int:
        """Remove all expired entries. Returns count of removed items."""
        now = time.time()
        expired = [k for k, (_, exp) in self._store.items() if now > exp]
        for k in expired:
            del self._store[k]
        if expired:
            logger.debug(f"Cache cleanup: removed {len(expired)} expired entries")
        return len(expired)

    @property
    def size(self) -> int:
        return len(self._store)

    def has(self, key: str) -> bool:
        return self.get(key) is not None


# ── Module-level singleton ────────────────────────────────────────────────────

cache = InMemoryCache()
