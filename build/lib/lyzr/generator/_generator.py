import os
from datetime import datetime
from typing import Optional, Union, Any

import pandas as pd

from lyzr.base.errors import InvalidModelError, check_values
from lyzr.base.file_utils import read_file, describe_dataset
from lyzr.base.llms import LLM, Prompt, get_model


class Generator:
    def __init__(self) -> None:
        return None

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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
