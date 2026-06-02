"""Domain abstraction + registry — replaces string dispatch scattered across the old code
(DOMAIN_STOP_TOKEN, DOMAIN_MAX_*, create_*_prompt, ModelHandler.is_stop, the vocab if/elif,
and the trainer's metric if/elif). One class per domain, registered once.
"""

from .base import Domain
from .registry import get_domain, register_domain, registered_domains

__all__ = ["Domain", "get_domain", "register_domain", "registered_domains"]
