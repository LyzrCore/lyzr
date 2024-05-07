# standart-library imports
import io
import time
import string
import logging
import hashlib
import traceback
from typing import Union
from functools import wraps

# third-party imports
import numpy as np
import pandas as pd

# local imports
from lyzr.base import SystemMessage, AssistantMessage, LiteLLM
from lyzr.base.errors import AnalysisFailedError


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
        clean_col = remove_punctuation_from_string(col)
        if clean_col in columns_dict:
            column_names.append(columns_dict[clean_col])
    return column_names


def remove_punctuation_from_string(value: str) -> str:
    value = str(value).strip()
    value = value.translate(str.maketrans("", "", string.punctuation))
    value = value.replace(" ", "").lower()
    return value


def get_info_dict_from_df_dict(df_dict: dict[pd.DataFrame]) -> dict[str]:
    df_info_dict = {}
    for name, df in df_dict.items():
        if not isinstance(df, pd.DataFrame):
            continue
        buffer = io.StringIO()
        df.info(buf=buffer)
        df_info_dict[name] = buffer.getvalue()
    return df_info_dict


def format_df_with_info(df_dict: dict[pd.DataFrame]) -> str:
    df_info_dict = get_info_dict_from_df_dict(df_dict)
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


def format_df_details(output_df, name: str = None) -> str:
    if isinstance(output_df, pd.Series):
        output_df = output_df.to_frame()
    if isinstance(output_df, list):
        return "\n".join([df_details_with_describe(df) for df in output_df])
    if isinstance(output_df, dict):
        return "\n".join(
            [df_details_with_describe(df, name) for name, df in output_df.items()]
        )
    if not isinstance(output_df, pd.DataFrame):
        return str(output_df)
    else:
        return df_details_with_describe(output_df, name)


def df_details_with_describe(output_df, name: str = None) -> str:
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


def logging_decorator(logger: logging.Logger):
    def decorator_wrapper(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger.info(
                f"Starting {func.__name__}",
                extra={
                    "function": func.__name__,
                    "input_args": args,
                    "input_kwargs": kwargs,
                },
            )
            try:
                result = func(*args, **kwargs)
                logger.info(
                    f"Completed {func.__name__}",
                    extra={
                        "function": func.__name__,
                        "input_args": args,
                        "input_kwargs": kwargs,
                        "output": result,
                    },
                )
            except Exception as e:
                logger.error(
                    f"Error in {func.__name__}: {e.__class__.__name__}.",
                    extra={
                        "function": func.__name__,
                        "traceback": traceback.format_exc().splitlines(),
                    },
                )
                raise e
            return result

        return wrapper

    return decorator_wrapper


def iterate_llm_calls(
    max_retries=1,
    *,
    llm: LiteLLM,
    llm_messages: list,
    logger: logging.Logger,
    log_messages: dict = {},
    time_limit: int = 30,
    llm_kwargs: dict = {},
):
    def decorator_wrapper(func):
        def wrapped_func(**kwargs):
            result = None
            logger.info(log_messages.get("start", "Starting LLM analysis."))
            start_time = time.time()
            result = repeater(
                max_retries=max_retries,
                start_time=start_time,
                time_limit=time_limit,
                logger=logger,
                llm=llm,
                llm_messages=llm_messages,
                func=func,
                llm_kwargs=llm_kwargs,
                **kwargs,
            )
            if result is None:
                logger.info("Result is None")
            logger.info(log_messages.get("end", "LLM analysis completed."))
            return result

        return wrapped_func

    return decorator_wrapper


def repeater(
    max_retries: int,
    start_time: float,
    time_limit: int,
    logger: logging.Logger,
    llm: LiteLLM,
    llm_messages: list,
    func: callable,
    llm_kwargs: dict = {},
    **kwargs,
):
    for i in range(max_retries):
        try:
            llm_response = llm.run(messages=llm_messages, **llm_kwargs)
        except Exception as e:
            logger.error(
                f"Error with getting response from LLM in try {i + 1}. {e.__class__.__name__}: {e}. Traceback: {traceback.format_exc()}"
            )
            continue
        try:
            result = func(llm_response=llm_response.message.content, **kwargs)
            if result is not None:
                logger.info(f"Result recieved: {result}")
                break
        except Exception as e:
            logger.error(
                f"Error in try {i + 1}. {e.__class__.__name__}: {e}. Traceback: {traceback.format_exc()}"
            )
            llm_messages.append(AssistantMessage(content=llm_response.message.content))
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
