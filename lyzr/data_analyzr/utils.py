# standart-library imports
import io
import re
import time
import string
import logging
import hashlib
import traceback
from typing import Union

# third-party imports
import numpy as np
import pandas as pd

# local imports
from lyzr.base.errors import AnalysisFailedError
from lyzr.base import SystemMessage, AssistantMessage, LiteLLM


def deterministic_uuid(content: Union[str, bytes, list] = None):
    if content is None:
        content = str(time.time())
    if isinstance(content, list):
        content = "/".join([str(x) for x in content if x is not None])
    if isinstance(content, str):
        content = content.encode("utf-8")
    hash_object = hashlib.md5(content)
    return hash_object.hexdigest()


def get_columns_names(
    df_columns: pd.DataFrame.columns,
    arguments: dict = {},
    columns: list = None,
    logger: logging.Logger = None,
) -> list:
    if isinstance(df_columns, pd.MultiIndex):
        df_columns = df_columns.levels[0]
    if columns is None:
        if "columns" not in arguments or not isinstance(arguments.get("columns"), list):
            return df_columns.to_list()
        columns = arguments.get("columns", [])
    if len(columns) == 0:
        return df_columns.to_list()
    columns_dict = {remove_punctuation_from_string(col): col for col in df_columns}
    column_names = []
    for col in columns:
        if remove_punctuation_from_string(col) not in columns_dict:
            logger.warning(
                "Invalid column name provided: {}. Skipping this column.".format(col)
            )
        else:
            column_names.append(columns_dict[remove_punctuation_from_string(col)])
    return column_names


def remove_punctuation_from_string(value: str) -> str:
    value = str(value).strip()
    value = value.translate(str.maketrans("", "", string.punctuation))
    value = value.replace(" ", "").lower()
    return value


def flatten_list(lst: list):
    return_lst = []
    for el in lst:
        if isinstance(el, list):
            return_lst.extend(flatten_list(el))
        else:
            return_lst.append(el)
    return return_lst


def _remove_punctuation_from_numeric_string(value) -> str:
    if not isinstance(value, str):
        return value
    negative = False
    value = value.strip()
    if value[0] == "-":
        negative = True
    cleaned = re.sub(r"[^\d.]", "", str(value))
    if cleaned.replace(".", "").isdigit():
        return "-" + cleaned if negative else cleaned
    else:
        return value


def convert_to_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for col in columns:
        try:
            df = df.dropna(subset=[col])
            column_values = df.loc[:, col].apply(remove_punctuation_from_string)
            column_values = pd.to_numeric(column_values)
            df.loc[:, col] = column_values.astype("float")
        except Exception:
            pass
    return df.infer_objects()


def get_info_dict_from_df_dict(df_dict: dict[pd.DataFrame]) -> dict[str]:
    df_info_dict = {}
    for name, df in df_dict.items():
        if not isinstance(df, pd.DataFrame):
            continue
        buffer = io.StringIO()
        df.info(buf=buffer)
        df_info_dict[name] = buffer.getvalue()
    return df_info_dict


def format_df_details(df_dict: dict[pd.DataFrame], df_info_dict: dict[str]) -> str:
    str_output = []
    for name, df in df_dict.items():
        var_name = name.lower().replace(" ", "_")
        if name in df_info_dict and isinstance(df, pd.DataFrame):
            str_output.append(
                f"Dataframe: `{var_name}`\nOutput of `{var_name}.head()`:\n{df.head()}\n\nOutput of `{var_name}.info()`:\n{df_info_dict[name]}\n"
            )
        else:
            str_output.append(f"{name}:\n{df}\n")
    return "\n".join(str_output)


def format_df_with_info(df_dict: dict[pd.DataFrame]) -> str:
    return format_df_details(df_dict, get_info_dict_from_df_dict(df_dict))


def format_df_with_describe(output_df, name: str = None) -> str:
    if isinstance(output_df, pd.Series):
        output_df = output_df.to_frame()
    if isinstance(output_df, list):
        return "\n".join([format_df_with_describe(df) for df in output_df])
    if isinstance(output_df, dict):
        return "\n".join(
            [format_df_with_describe(df, name) for name, df in output_df.items()]
        )
    if not isinstance(output_df, pd.DataFrame):
        return str(output_df)

    name = name or "Dataframe"
    if output_df.size > 100:
        df_display = pd.concat([output_df.head(50), output_df.tail(50)], axis=0)
        df_string = f"{name} snapshot:\n{_df_to_string(df_display)}\n\nOutput of `df.describe()`:\n{_df_to_string(output_df.describe())}"
    else:
        df_string = f"{name}:\n{_df_to_string(output_df)}"
    return df_string


def _df_to_string(output_df: pd.DataFrame) -> str:
    output_df.columns = [str(col) for col in output_df.columns.tolist()]
    # convert all datetime columns to datetime objects
    datetimecols = [
        col
        for col in output_df.columns.tolist()
        if ("date" in col.lower() or "time" in col.lower())
        and output_df[col].dtype != np.number
    ]
    if "timestamp" in output_df.columns and "timestamp" not in datetimecols:
        datetimecols.append("timestamp")
    for col in datetimecols:
        output_df[col] = output_df[col].astype(dtype="datetime64[ns]", errors="ignore")
        output_df.loc[:, col] = pd.to_datetime(output_df[col], errors="ignore")

    datetimecols = output_df.select_dtypes(include=["datetime64"]).columns.tolist()
    formatters = {col: _format_date for col in datetimecols}
    return output_df.to_string(
        float_format="{:,.2f}".format,
        formatters=formatters,
        na_rep="None",
    )


def _format_date(date: pd.Timestamp):
    return date.strftime("%d %b %Y %H:%M")


def iterate_llm_calls(
    max_tries=1,
    *,
    llm: LiteLLM,
    llm_messages: list,
    logger: logging.Logger,
    log_messages={},
    time_limit: int = 30,
):
    def decorator_wrapper(func):
        def wrapped_func(**kwargs):
            result = None
            logger.info(log_messages.get("start", "Starting LLM analysis."))
            start_time = time.time()
            result = repeater(
                max_tries=max_tries,
                start_time=start_time,
                time_limit=time_limit,
                logger=logger,
                llm=llm,
                llm_messages=llm_messages,
                func=func,
                **kwargs,
            )
            if result is None:
                logger.info("Result is None")
            logger.info(log_messages.get("end", "LLM analysis completed."))
            return result

        return wrapped_func

    return decorator_wrapper


def repeater(
    max_tries: int,
    start_time: float,
    time_limit: int,
    logger: logging.Logger,
    llm: LiteLLM,
    llm_messages: list,
    func: callable,
    **kwargs,
):
    for i in range(max_tries):
        try:
            llm_response = llm.run(messages=llm_messages)
        except Exception as e:
            logger.error(
                f"Error with getting response from LLM in try {i + 1}. {e.__class__.__name__}: {e}. Traceback: {traceback.format_exc()}"
            )
            continue
        try:
            result = func(llm_response=llm_response, **kwargs)
            if result is not None:
                logger.info(f"Result recieved: {result}")
                break
        except Exception as e:
            logger.error(
                f"Error in try {i + 1}. {e.__class__.__name__}: {e}. Traceback: {traceback.format_exc()}"
            )
            llm_messages.append(AssistantMessage(content=llm_response))
            llm_messages.append(
                SystemMessage(
                    content="Your response resulted in the following error:\n"
                    f"{e.__class__.__name__}: {e}\n{traceback.format_exc()}\n\n"
                    "Please correct your response to prevent this error."
                )
            )
        finally:
            if time.time() - start_time > time_limit and time_limit > 0:
                raise AnalysisFailedError(
                    "The request could not be completed. Please wait a while and try again."
                )
    return result
