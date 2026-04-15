"""Activation cache module for computational state persistence.

Caches neuron activation states at session boundaries to enable
warm-start recall, reducing latency by skipping activation recomputation.

Inspired by "Memory Caching: RNNs with Growing Memory" (ICLR 2026).
"""

from neural_memory.cache.manager import ActivationCacheManager
from neural_memory.cache.models import ActivationCache, CachedState

__all__ = ["ActivationCache", "CachedState", "ActivationCacheManager"]
