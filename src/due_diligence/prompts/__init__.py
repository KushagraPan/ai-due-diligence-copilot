"""
Prompt templates and prompt formatting utilities for AI Due Diligence Co-pilot.
"""

from due_diligence.prompts.generation import (
    GENERATION_SYSTEM_PROMPT,
    format_generation_user_prompt,
)
from due_diligence.prompts.query_rewriter import (
    QUERY_REWRITER_SYSTEM_PROMPT,
    QUERY_REWRITER_TAG,
    format_query_rewriter_user_prompt,
)
from due_diligence.prompts.reranker import (
    LLM_RERANKER_SYSTEM_PROMPT,
    format_llm_reranker_prompt,
)

__all__ = [
    "GENERATION_SYSTEM_PROMPT",
    "LLM_RERANKER_SYSTEM_PROMPT",
    "QUERY_REWRITER_SYSTEM_PROMPT",
    "QUERY_REWRITER_TAG",
    "format_generation_user_prompt",
    "format_llm_reranker_prompt",
    "format_query_rewriter_user_prompt",
]

