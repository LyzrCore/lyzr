import os
from datetime import datetime
from typing import Optional, Union

import pandas as pd

from lyzr.base.errors import InvalidModelError, check_values
from lyzr.base.file_utils import read_file
from lyzr.base.llms import LLM, Prompt, get_model


def insights(
    model: Optional[LLM] = None,
    query: Optional[str] = None,
    api_key: Optional[str] = None,
    model_type: Optional[str] = None,
    model_name: Optional[str] = None,
    description: Optional[str] = None,
    file_kwargs: Optional[dict] = None,
    prompt: Optional[Union[Prompt, str]] = None,
    result: Optional[Union[pd.DataFrame, pd.Series, str]] = None,
    **kwargs,
) -> str:
    prompt = prompt or os.getenv("PROMPT") or "insights_pt"
    result = result if result is not None else os.getenv("RESULT")

    check_values(query=query, model=model, params={"result": result})

    if model is None:
        model = get_model(api_key, model_type, model_name)
    elif not isinstance(model, LLM):
        raise InvalidModelError

    if isinstance(result, str):
        result = read_file(result, **file_kwargs)

    model.prompt = prompt if isinstance(prompt, Prompt) else Prompt(prompt)
    if model.prompt.get_variables() != []:
        model.set_prompt(
            date=datetime.now().strftime("%Y-%m-%d"),
            description=description,
            result=result,
            query=query,
            **kwargs,
        )

    output = model.run()

    return output["choices"][0]["message"]["content"]
