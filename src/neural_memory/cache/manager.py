"""Activation cache manager for session-level caching.

Handles save/load/invalidation of cached activation states.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from neural_memory.cache.models import ActivationCache, CachedState
from neural_memory.cache.serializer import (
    cache_exists,
    delete_cache,
    load_cache,
    save_cache,
)
from neural_memory.utils.timeutils import utcnow

if TYPE_CHECKING:
    from neural_memory.storage.base import NeuralStorage

logger = logging.getLogger(__name__)

# Default limits
DEFAULT_MAX_ENTRIES = 200
DEFAULT_TTL_HOURS = 24
DEFAULT_MIN_ACTIVATION = 0.1


class ActivationCacheManager:
    """Manages activation state caching for warm-start recall.

    Usage:
        manager = ActivationCacheManager(storage)

        # Save at session end
        await manager.save_snapshot()

        # Load at session start
        states = await manager.load_snapshot()

        # Check validity
        if manager.is_cache_valid():
            # Use cached states
            ...
    """

    def __init__(
        self,
        storage: NeuralStorage,
        data_dir: Path | None = None,
        max_entries: int = DEFAULT_MAX_ENTRIES,
        ttl_hours: int = DEFAULT_TTL_HOURS,
        min_activation: float = DEFAULT_MIN_ACTIVATION,
    ) -> None:
        """Initialize cache manager.

        Args:
            storage: Neural storage backend
            data_dir: Data directory for cache files
            max_entries: Max neurons to cache
            ttl_hours: Cache time-to-live
            min_activation: Minimum activation level to cache
        """
        self._storage = storage
        self._data_dir = data_dir
        self._max_entries = max_entries
        self._ttl_hours = ttl_hours
        self._min_activation = min_activation
        self._loaded_cache: ActivationCache | None = None
        self._hit_count = 0
        self._miss_count = 0

    @property
    def brain_id(self) -> str:
        """Get current brain ID from storage."""
        return getattr(self._storage, "brain_id", "") or ""

    @property
    def brain_name(self) -> str:
        """Get current brain name from storage."""
        brain_id = self.brain_id
        if not brain_id:
            return "default"
        # Try to get brain name from storage
        # This is sync access to cached brain info
        return getattr(self._storage, "_brain_name", None) or "default"

    async def save_snapshot(self) -> ActivationCache | None:
        """Save current activation states to cache.

        Captures top-N neurons by activation level and persists to disk.

        Returns:
            ActivationCache if saved, None on error
        """
        brain_id = self.brain_id
        if not brain_id:
            logger.debug("No brain_id, skipping activation cache save")
            return None

        try:
            brain = await self._storage.get_brain(brain_id)
            brain_name = brain.name if brain else "default"

            # Get all neuron states
            all_states = await self._get_top_states()

            if not all_states:
                logger.debug("No active neurons to cache")
                return None

            # Compute brain hash for staleness detection
            brain_hash = await self._compute_brain_hash()

            now = utcnow()
            cache = ActivationCache(
                brain_id=brain_id,
                brain_name=brain_name,
                cached_at=now,
                ttl_hours=self._ttl_hours,
                entries=tuple(all_states),
                brain_hash=brain_hash,
            )

            save_cache(cache, self._data_dir)
            self._loaded_cache = cache

            logger.info(
                "Saved activation cache: %d entries for brain '%s'",
                len(all_states),
                brain_name,
            )
            return cache

        except Exception as e:
            logger.error("Failed to save activation cache: %s", e, exc_info=True)
            return None

    async def load_snapshot(self) -> list[CachedState]:
        """Load cached activation states if valid.

        Returns:
            List of cached states, empty if cache invalid/missing
        """
        # Fetch brain name consistently with save_snapshot
        brain_id = self.brain_id
        if not brain_id:
            self._miss_count += 1
            return []

        brain = await self._storage.get_brain(brain_id)
        brain_name = brain.name if brain else "default"

        cache = load_cache(brain_name, self._data_dir)
        if cache is None:
            self._miss_count += 1
            return []

        # Check expiry
        if cache.is_expired():
            logger.debug("Activation cache expired for '%s'", brain_name)
            self._miss_count += 1
            return []

        # Check brain hash (staleness)
        current_hash = await self._compute_brain_hash()
        if cache.brain_hash and cache.brain_hash != current_hash:
            logger.debug(
                "Activation cache stale for '%s' (hash mismatch)",
                brain_name,
            )
            self._miss_count += 1
            return []

        self._loaded_cache = cache
        self._hit_count += 1

        logger.info(
            "Loaded activation cache: %d entries for brain '%s'",
            cache.entry_count,
            brain_name,
        )
        return list(cache.entries)

    def is_cache_valid(self) -> bool:
        """Check if current loaded cache is valid."""
        if self._loaded_cache is None:
            return False
        return not self._loaded_cache.is_expired()

    def get_cached_state(self, neuron_id: str) -> CachedState | None:
        """Get cached state for a neuron.

        Args:
            neuron_id: Neuron UUID

        Returns:
            CachedState if found in cache, None otherwise
        """
        if self._loaded_cache is None:
            self._miss_count += 1
            return None

        state = self._loaded_cache.get_state(neuron_id)
        if state:
            self._hit_count += 1
        else:
            self._miss_count += 1
        return state

    async def invalidate(self) -> bool:
        """Invalidate and delete current cache.

        Call this when brain state changes (new neurons, synapses, etc.)

        Returns:
            True if cache was deleted
        """
        self._loaded_cache = None
        return delete_cache(self.brain_name, self._data_dir)

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dict with hit_count, miss_count, hit_rate, entries, age_seconds
        """
        total = self._hit_count + self._miss_count
        hit_rate = self._hit_count / total if total > 0 else 0.0

        stats: dict[str, Any] = {
            "hit_count": self._hit_count,
            "miss_count": self._miss_count,
            "hit_rate": round(hit_rate, 3),
            "entries": 0,
            "age_seconds": 0,
            "brain_name": self.brain_name,
            "cache_exists": cache_exists(self.brain_name, self._data_dir),
        }

        if self._loaded_cache:
            stats["entries"] = self._loaded_cache.entry_count
            age = (utcnow() - self._loaded_cache.cached_at).total_seconds()
            stats["age_seconds"] = int(age)
            stats["ttl_hours"] = self._loaded_cache.ttl_hours

        return stats

    async def _get_top_states(self) -> list[CachedState]:
        """Get top neuron states by activation level."""
        # Get all neuron IDs
        all_neurons = await self._storage.find_neurons(limit=1000)
        if not all_neurons:
            return []

        neuron_ids = [n.id for n in all_neurons]
        states_map = await self._storage.get_neuron_states_batch(neuron_ids)

        # Filter and sort by activation
        cached: list[CachedState] = []
        for nid, state in states_map.items():
            if state.activation_level >= self._min_activation:
                cached.append(
                    CachedState(
                        neuron_id=nid,
                        activation_level=state.activation_level,
                        access_frequency=state.access_frequency,
                        last_activated=state.last_activated,
                    )
                )

        # Sort by activation descending, take top N
        cached.sort(key=lambda x: x.activation_level, reverse=True)
        return cached[: self._max_entries]

    async def _compute_brain_hash(self) -> str:
        """Compute hash of brain state for staleness detection.

        Uses neuron count + synapse count + fiber count + last modified timestamp.
        This provides reasonable staleness detection for most operations.
        """
        brain_id = self.brain_id
        if not brain_id:
            return ""

        try:
            stats = await self._storage.get_stats(brain_id)
            neuron_count = stats.get("neuron_count", 0)
            synapse_count = stats.get("synapse_count", 0)
            fiber_count = stats.get("fiber_count", 0)
            # Include last_modified if available for better staleness detection
            last_modified = stats.get("last_modified", "")

            # Hash includes counts + timestamp for better staleness detection
            data = f"{brain_id}:{neuron_count}:{synapse_count}:{fiber_count}:{last_modified}"
            return hashlib.md5(data.encode()).hexdigest()[:16]
        except Exception:
            return ""
