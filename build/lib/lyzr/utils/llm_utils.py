import logging

from lyzr.utils.templates import QUERY_PROMPT_TEMPLATE, SYSTEM_PROMPT

logger = logging.getLogger(__name__)


def set_default_prompt_template():
    return SYSTEM_PROMPT, QUERY_PROMPT_TEMPLATE
