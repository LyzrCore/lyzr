# standard imports
import os
import string
import logging

# third-party imports
import pandas as pd

# local imports
from lyzr.data_analyzr.utils import deterministic_uuid


def handle_plotpath(plot_path: str, output_format: str, logger: logging.Logger) -> str:
    fixed_path = _fix_plotpath(plot_path, output_format)
    try:
        open(fixed_path, "w").close()
        return fixed_path
    except Exception:
        random_path = os.path.join(
            "generated_plots", f"{deterministic_uuid()}.{output_format}"
        )
        logger.warning(
            f"Incorrect path for plot image provided: {plot_path}. Using {random_path} instead.",
            extras={"function": "handle_plotpath"},
        )
        return handle_plotpath(random_path, output_format, logger)


def _fix_plotpath(plot_path: str, output_format: str) -> str:
    if os.path.isdir(plot_path):
        plot_path = os.path.join(plot_path, f"plot.{output_format}")
    if os.path.splitext(plot_path)[1] != f".{output_format}":
        plot_path = os.path.splitext(plot_path)[0] + f".{output_format}"
    dir_path = os.path.dirname(plot_path)
    if dir_path.strip() != "":
        os.makedirs(dir_path, exist_ok=True)
    return plot_path


def set_analysis_attribute(analysis_output, analyzer):
    if analyzer is None:
        return False
    if (not isinstance(analysis_output, pd.DataFrame)) or (analysis_output.size < 2):
        return True
    else:
        return False


def flatten_list(lst: list):
    return_lst = []
    for el in lst:
        if isinstance(el, list):
            return_lst.extend(flatten_list(el))
        elif el is not None:
            return_lst.append(el)
    return return_lst


def _remove_punctuation_from_numeric_string(value) -> str:
    if not isinstance(value, str):
        return value
    negative = False
    value = value.strip()
    if value[0] == "-":
        negative = True
    punctuation_other_than_dot = "".join(
        [char for char in string.punctuation if char != "."]
    )
    cleaned = value.translate(str.maketrans("", "", punctuation_other_than_dot))
    if cleaned.replace(".", "").isdigit():
        return "-" + cleaned if negative else cleaned
    else:
        return value


def convert_to_numeric(df: pd.DataFrame, column: str) -> pd.DataFrame:
    df = df.dropna(subset=[column])
    column_values = df.loc[:, column].apply(_remove_punctuation_from_numeric_string)
    column_values = pd.to_numeric(column_values)
    df.loc[:, column] = column_values.astype("float")
    return df


def convert_to_categorical(df: pd.DataFrame, column: str) -> pd.DataFrame:
    df.loc[:, column] = df.loc[:, column].astype("category")
    return df


def convert_to_datetime(df: pd.DataFrame, columns: str) -> pd.DataFrame:
    df.loc[:, columns] = df.loc[:, columns].apply(pd.to_datetime, errors="coerce")
    return df


def convert_to_bool(df: pd.DataFrame, columns: str) -> pd.DataFrame:
    df.loc[:, columns] = df.loc[:, columns].astype(bool)
    return df


def convert_to_string(df: pd.DataFrame, columns: str) -> pd.DataFrame:
    df.loc[:, columns] = df.loc[:, columns].astype(str)
    return df


def convert_dtypes(df: pd.DataFrame, columns: dict[str, str]) -> pd.DataFrame:
    for col, dtype in columns.items():
        try:
            if col not in df.columns:
                continue
            if dtype == "datetime":
                df = convert_to_datetime(df, col)
            elif dtype == "numeric":
                df = convert_to_numeric(df, col)
            elif dtype == "categorical":
                df = convert_to_categorical(df, col)
            elif dtype == "bool":
                df = convert_to_bool(df, col)
            elif dtype == "string":
                df = convert_to_string(df, col)
            elif dtype == "object":
                df.loc[:, col] = df.loc[:, col].astype(object)
            else:
                df.loc[:, col] = df.loc[:, col].astype(object)
        except Exception:
            pass
    return df
