import os
import pickle
from typing import Optional

import pandas as pd

from lyzr.base.llm import LiteLLM, LyzrLLMFactory
from lyzr.base.prompt import LyzrPromptFactory


def read_file(
    filepath: str, encoding: Optional[str] = "utf-8", **kwargs
) -> pd.DataFrame:
    if not os.path.exists(filepath):
        raise ValueError(
            f"File '{filepath}' not found. Please provide a valid filepath."
        )
    file_extension = filepath.split(".")[-1]
    try:
        if file_extension == "csv":
            return pd.read_csv(filepath, encoding=encoding, **kwargs)
        elif file_extension == "tsv":
            return pd.read_csv(filepath, sep="\t", encoding=encoding, **kwargs)
        elif file_extension == "txt":
            with open(filepath, "r") as f:
                return f.read(encoding=encoding, **kwargs)
        elif file_extension == "json":
            return pd.read_json(filepath, encoding=encoding, **kwargs)
        elif file_extension in ["xlsx", "xls"]:
            return pd.read_excel(filepath, encoding=encoding, **kwargs)
        elif file_extension == "pkl":
            with open(filepath, "rb") as f:
                return pickle.load(f, encoding=encoding, **kwargs)
        else:
            raise ValueError(
                f"File extension '{file_extension}' not supported. Please provide a csv or pkl file."
            )
    except FileNotFoundError:
        raise FileNotFoundError(
            f"File '{filepath}' not found. Please provide a valid filepath."
        )
    except UnicodeDecodeError:
        raise UnicodeDecodeError(
            f"File '{filepath}' could not be decoded. Please provide a file with utf-8 encoding."
            "If the file is not encoded in utf-8, please provide the encoding as a parameter: file_kwargs={'encoding': 'utf-8'}"
        )


def describe_dataset(
    model: Optional[LiteLLM] = None,
    df: Optional[pd.DataFrame] = None,
    api_key: Optional[str] = None,
    model_type: Optional[str] = None,
    model_name: Optional[str] = None,
) -> str:
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Please provide a valid pandas DataFrame.")

    if model is None:
        model = LyzrLLMFactory.from_defaults(
            api_key=api_key,
            api_type=model_type,
            model=model_name,
        )

    messages = [
        LyzrPromptFactory(
            name="dataset_description", prompt_type="system"
        ).get_message(),
        LyzrPromptFactory(name="dataset_description", prompt_type="user").get_message(
            headers=df.columns.tolist(), df_sample=df.head()
        ),
    ]
    output = model.run(messages=messages)

    return output.message.content
