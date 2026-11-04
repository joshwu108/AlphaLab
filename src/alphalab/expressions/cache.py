"""Bounded LRU cache of evaluated sub-expressions.

Search algorithms evaluate thousands of expressions that share sub-trees
(``ts_mean($close, 20)`` appears in many candidates), so caching every evaluated
node avoids recomputing shared work.

Keys are ``(panel fingerprint, canonical expression)``. Because the fingerprint
covers every value and timestamp of the panel, a cached array can only be
reused for byte-identical data; a different or modified panel misses the cache
automatically. Cached arrays are read-only, so a caller cannot corrupt an entry.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

CacheKey = tuple[str, str]

DEFAULT_MAX_BYTES = 512 * 1024 * 1024


@dataclass(frozen=True)
class CacheStats:
    """Counters since the cache was created or last cleared."""

    hits: int
    misses: int
    evictions: int
    entries: int
    bytes_used: int

    @property
    def hit_rate(self) -> float:
        """Fraction of lookups served from the cache (0 when there were none)."""
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


class ExpressionCache:
    """LRU cache bounded by total array bytes."""

    def __init__(self, max_bytes: int = DEFAULT_MAX_BYTES) -> None:
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        self.max_bytes = max_bytes
        self._entries: OrderedDict[CacheKey, np.ndarray] = OrderedDict()
        self._bytes = 0
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def get(self, key: CacheKey) -> np.ndarray | None:
        """Return the cached array or ``None``; a hit refreshes recency."""
        array = self._entries.get(key)
        if array is None:
            self._misses += 1
            return None
        self._entries.move_to_end(key)
        self._hits += 1
        return array

    def put(self, key: CacheKey, array: np.ndarray) -> np.ndarray:
        """Store a read-only copy of ``array`` and return it.

        Arrays larger than ``max_bytes`` are returned without being stored.
        """
        stored = np.array(array, dtype=np.float64, copy=True)
        stored.flags.writeable = False
        if stored.nbytes > self.max_bytes:
            return stored
        previous = self._entries.pop(key, None)
        if previous is not None:
            self._bytes -= previous.nbytes
        self._entries[key] = stored
        self._bytes += stored.nbytes
        while self._bytes > self.max_bytes:
            _, evicted = self._entries.popitem(last=False)
            self._bytes -= evicted.nbytes
            self._evictions += 1
        return stored

    def invalidate(self, fingerprint: str | None = None) -> int:
        """Drop entries for one panel fingerprint (or all); return how many were dropped."""
        if fingerprint is None:
            dropped = len(self._entries)
            self._entries.clear()
            self._bytes = 0
            return dropped
        doomed = [k for k in self._entries if k[0] == fingerprint]
        for key in doomed:
            self._bytes -= self._entries.pop(key).nbytes
        return len(doomed)

    def clear(self) -> None:
        """Drop all entries and reset counters."""
        self.invalidate()
        self._hits = self._misses = self._evictions = 0

    def __contains__(self, key: object) -> bool:
        return key in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def stats(self) -> CacheStats:
        """Current counters."""
        return CacheStats(
            self._hits, self._misses, self._evictions, len(self._entries), self._bytes
        )
