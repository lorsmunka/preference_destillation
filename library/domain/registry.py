"""Domain registry — `get_domain(name)`. Add a domain = new file + `register_domain`."""

from typing import Dict, List, Union

from .base import Domain
from .math_word_problem import MathWordProblemDomain
from .post_generation import PostGenerationDomain
from .reddit_comment_sentiment import RedditSentimentDomain

_REGISTRY: Dict[str, Domain] = {}


def register_domain(domain: Domain) -> None:
    _REGISTRY[domain.name] = domain


def get_domain(name: str) -> Domain:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown domain {name!r}. Registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def registered_domains() -> List[str]:
    return sorted(_REGISTRY)


def as_domain(value: Union[str, Domain]) -> Domain:
    """Normalize a domain reference to a Domain instance, with no side effects: a Domain
    is returned as-is (used directly when running), a name string is looked up in the
    registry. A custom domain only needs `register_domain` for the name-based steps that
    run in a separate process (build-vocab, analysis) — not to train or generate data."""
    if isinstance(value, Domain):
        return value
    return get_domain(value)


for _domain in (RedditSentimentDomain(), MathWordProblemDomain(), PostGenerationDomain()):
    register_domain(_domain)
