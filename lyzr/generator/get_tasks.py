import os
from datetime import datetime
from typing import Optional, Union

from lyzr.base.errors import InvalidModelError, check_values
from lyzr.base.llms import LLM, Prompt, get_model


def tasks(
    model: Optional[LLM] = None,
    query: Optional[str] = None,
    insights: Optional[str] = None,
    recommendations: Optional[str] = None,
    model_type: Optional[str] = None,
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
    description: Optional[str] = None,
    prompt: Optional[Union[Prompt, str]] = None,
    **kwargs,
) -> str:
    prompt = prompt or os.getenv("PROMPT") or "tasks_pt"

    check_values(
        query=query,
        model=model,
        params=dict(insights=insights, recommendations=recommendations),
    )

    if model is None:
        model = get_model(api_key, model_type, model_name)
    elif not isinstance(model, LLM):
        raise InvalidModelError

    model.prompt = prompt if isinstance(prompt, Prompt) else Prompt(prompt)
    if model.prompt.get_variables() != []:
        model.set_prompt(
            prompt_name=prompt,
            date=datetime.now().strftime("%Y-%m-%d"),
            recommendations=recommendations,
            description=description,
            insights=insights,
            query=query,
            **kwargs,
        )

    output = model.run()

    return output["choices"][0]["message"]["content"]
