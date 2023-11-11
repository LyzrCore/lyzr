import os
from datetime import datetime
from typing import Optional, Union, Any

from lyzr.base.errors import InvalidModelError, check_values
from lyzr.base.llms import LLM, Prompt, get_model


def recommendations(
    model: Optional[LLM] = None,
    query: Optional[str] = None,
    insights: Optional[str] = None,
    model_type: Optional[str] = None,
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
    schema: Optional[Any] = None,
    description: Optional[str] = None,
    prompt: Optional[Union[Prompt, str]] = None,
    **kwargs,
) -> str:
    prompt = prompt or os.getenv("PROMPT") or "recommendations_pt"

    check_values(query=query, model=model, params={"insights": insights})

    if model is None:
        model = get_model(api_key, model_type, model_name)
    if not isinstance(model, LLM):
        raise InvalidModelError

    schema = schema or [
        {
            "Recommendation": "string",
            "Basis of the Recommendation": "string",
            "Impact if implemented": "string",
        },
        {
            "Recommendation": "string",
            "Basis of the Recommendation": "string",
            "Impact if implemented": "string",
        },
        {
            "Recommendation": "string",
            "Basis of the Recommendation": "string",
            "Impact if implemented": "string",
        },
    ]

    model.prompt = prompt if isinstance(prompt, Prompt) else Prompt(prompt)
    if model.prompt.get_variables() != []:
        model.set_prompt(
            date=datetime.now().strftime("%Y-%m-%d"),
            description=description,
            insights=insights,
            schema=schema,
            query=query,
            **kwargs,
        )

    output = model.run()

    return output["choices"][0]["message"]["content"]
