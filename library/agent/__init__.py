"""Parts of an agent that do not depend on the domain. question_templates: a pupil's (user's)
messages written once as templates and filled in from what a conversation shows."""

from .question_templates import Focus, Message, QuestionGenerator, Template, load_bank
