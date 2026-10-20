"""Domain registry — `get_domain(name)`. Add a domain = new file + `register_domain`."""

from typing import Dict, List, Union

from .base import Domain
from .code_recall import CodeRecallDomain
from .math_tutor import MathTutorDomain
from .math_tutor.agentic import ANNOTATIONS, TEACHER_MODES, MathTutorAgenticDomain
from .long_context_recall import LongContextRecallDomain
from .math_word_problem import MathWordProblemDomain
from .post_generation import PostGenerationDomain
from .reddit_comment_with_feedback import RedditCommentWithFeedbackDomain
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


for _domain in (RedditSentimentDomain(), MathWordProblemDomain(), PostGenerationDomain(), RedditCommentWithFeedbackDomain(), LongContextRecallDomain(), CodeRecallDomain(), MathTutorDomain()):
    register_domain(_domain)

# The agentic tutor: one registered variant per annotation format and teacher mode.
for _annotation in ANNOTATIONS:
    for _teacher in TEACHER_MODES:
        register_domain(MathTutorAgenticDomain(_annotation, _teacher))

# The trial slice (math_tutor/slice.py): its own name, so its own batches folder.
from .math_tutor.slice import SliceDomain
register_domain(SliceDomain())

# The coding agent (code_agent/): a synthetic project in the context, find or explain.
from .code_agent import CodeAgentDomain
register_domain(CodeAgentDomain())
