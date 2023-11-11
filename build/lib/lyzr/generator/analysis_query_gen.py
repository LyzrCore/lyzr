import os
from datetime import datetime
from typing import Optional, Union, Any

import pandas as pd

from lyzr.base.errors import InvalidModelError
from lyzr.base.file_utils import read_file, describe_dataset
from lyzr.base.llms import LLM, Prompt, get_model


def ai_queries_df(
    model: Optional[LLM] = None,
    model_name: Optional[str] = None,
    model_type: Optional[str] = None,
    api_key: Optional[str] = None,
    df: Optional[pd.DataFrame] = None,
    description: Optional[str] = None,
    schema: Optional[Any] = None,
    prompt: Optional[Union[Prompt, str]] = None,
    file_kwargs: Optional[dict] = None,
    **kwargs,
) -> str:
    prompt = prompt or os.getenv("PROMPT") or "query_gen_pt"

    if model is None:
        model = get_model(api_key, model_type, model_name)
    elif not isinstance(model, LLM):
        raise InvalidModelError

    df = os.getenv("DATAFRAME") if df is None else df
    if not isinstance(df, pd.DataFrame):
        df = read_file(df, **file_kwargs)
    description = description or describe_dataset(model, df)

    schema = schema or [
        {
            "ai_query": "string",
            "visualization": "string",
            "rank": 1,
            "Problem Type": "string",
        },
        {
            "ai_query": "string",
            "visualization": "string",
            "rank": 2,
            "Problem Type": "string",
        },
    ]

    model.prompt = prompt if isinstance(prompt, Prompt) else Prompt(prompt)
    if model.prompt.get_variables() != []:
        model.set_prompt(
            date=datetime.now().strftime("%Y-%m-%d"),
            headers=df.columns.tolist(),
            df_sample=df.head(),
            description=description,
            schema=schema,
            **kwargs,
        )

    output = model.run()

    return output["choices"][0]["message"]["content"]
