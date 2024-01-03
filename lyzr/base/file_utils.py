import os
import pickle
from typing import Optional

import pandas as pd

from lyzr.base.errors import InvalidValueError
from lyzr.base.llms import LLM, get_model
from lyzr.base.prompt import Prompt


def read_file(
    filepath: str, encoding: Optional[str] = "utf-8", **kwargs
) -> pd.DataFrame:
    if not os.path.exists(filepath):
        raise InvalidValueError(["filepath", "pandas DataFrame"])
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
    model: Optional[LLM] = None,
    df: Optional[pd.DataFrame] = None,
    api_key: Optional[str] = None,
    model_type: Optional[str] = None,
    model_name: Optional[str] = None,
) -> str:
    if not isinstance(df, pd.DataFrame):
        raise InvalidValueError(["pandas DataFrame"])

    if model is None:
        model = get_model(api_key, model_type, model_name)

    model.prompt = Prompt("dataset_description_pt")
    if model.prompt.get_variables() != []:
        model.set_messages(
            headers=df.columns.tolist(),
            df_sample=df.head(),
        )

    output = model.run()

    return output["choices"][0]["message"]["content"]
