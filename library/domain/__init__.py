"""Domain abstraction + registry — replaces string dispatch scattered across the old code
(DOMAIN_STOP_TOKEN, DOMAIN_MAX_*, create_*_prompt, ModelHandler.is_stop, the vocab if/elif,
and the trainer's metric if/elif). One self-contained package per domain, registered once.

Pass a domain to a config as an object, not a magic string — either a ready singleton
(`MATH_WORD_PROBLEM`), a fresh instance (`RedditSentimentDomain()`), or your own subclass.
"""

from .base import Domain
from .registry import get_domain, register_domain, registered_domains, as_domain
from .reddit_comment_sentiment import RedditSentimentDomain
from .math_word_problem import MathWordProblemDomain
from .post_generation import PostGenerationDomain

# Registered singletons — the ergonomic, no-magic-string way to say `domain=...`.
REDDIT_SENTIMENT = get_domain("reddit_comment_sentiment")
MATH_WORD_PROBLEM = get_domain("math_word_problem")
POST_GENERATION = get_domain("post_generation")

__all__ = [
    "Domain",
    "get_domain",
    "register_domain",
    "registered_domains",
    "as_domain",
    "RedditSentimentDomain",
    "MathWordProblemDomain",
    "PostGenerationDomain",
    "REDDIT_SENTIMENT",
    "MATH_WORD_PROBLEM",
    "POST_GENERATION",
]
