"""Domain registry — `get_domain(name)`. Add a domain = new file + `register_domain`."""

from typing import Dict, List

from .base import Domain
from .math_word_problem import MathWordProblemDomain
from .post_generation import PostGenerationDomain
from .reddit_sentiment import RedditSentimentDomain

_REGISTRY: Dict[str, Domain] = {}


def register_domain(domain: Domain) -> None:
    _REGISTRY[domain.name] = domain


def get_domain(name: str) -> Domain:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown domain {name!r}. Registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def registered_domains() -> List[str]:
    return sorted(_REGISTRY)


for _domain in (RedditSentimentDomain(), MathWordProblemDomain(), PostGenerationDomain()):
    register_domain(_domain)
