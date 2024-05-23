# standart-library imports
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


def translate_string_name(name: str) -> str:
    punc = string.punctuation + " "
    return (
        name.lower().strip().translate(str.maketrans(punc, "_" * len(punc))).strip("_")
    )


def format_df_details(output_df, name: str = None) -> str:
    if isinstance(output_df, pd.Series):
        output_df = output_df.to_frame()
    if isinstance(output_df, list):
        return "\n".join([format_df_details(df) for df in output_df])
    if isinstance(output_df, dict):
        return "\n".join(
            [format_df_details(df, name) for name, df in output_df.items()]
        )
    if not isinstance(output_df, pd.DataFrame):
        return str(output_df)
    else:
        return df_details_with_describe(output_df, name)


def df_details_with_describe(output_df, name: str = None) -> str:
    name = name or "Dataframe"
    if output_df is None:
        return f"{name}: None"
    if output_df.size > 100:
        df_display = pd.concat([output_df.head(50), output_df.tail(50)], axis=0)
        df_string = f"DataFrame name: {name}\nDataFrame snapshot:\n{_df_to_string(df_display)}\n\nDataFrame column details:\n{_df_to_string(output_df.describe())}"
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
        and isinstance(output_df[col].dtype, np.number)
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


def get_context_dict(context_str: str, context_dict: dict = None):
    if context_dict is None:
        context_dict = {}
    context_dict["analysis"] = context_dict.get("analysis", context_str).strip()
    context_dict["visualisation"] = context_dict.get(
        "visualisation", context_str
    ).strip()
    context_dict["insights"] = context_dict.get("insights", context_str).strip()
    context_dict["recommendations"] = context_dict.get(
        "recommendations", context_str
    ).strip()
    context_dict["tasks"] = context_dict.get("tasks", context_str).strip()
    for key, value in context_dict.items():
        context_dict[key] = value + "\n\n" if value != "" else value.strip()
    return context_dict


def logging_decorator(logger: logging.Logger):
    def decorator_wrapper(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger.info(
                f"Starting {func.__name__}",
                extra={
                    "function": func.__name__,
                    "input_args": None if len(args) == 0 else args,
                    "input_kwargs": kwargs,
                },
            )
            try:
                result = func(*args, **kwargs)
                logger.info(
                    f"Completed {func.__name__}",
                    extra={
                        "function": func.__name__,
                        "input_args": None if len(args) == 0 else args,
                        "input_kwargs": kwargs,
                        "response": result,
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
    max_retries: int,
    *,
    llm: LiteLLM,
    llm_messages: list,
    logger: logging.Logger,
    log_messages: dict = {},
    time_limit: int = None,
    llm_kwargs: dict = {},
):
    def decorator_wrapper(func):
        def wrapped_func(**kwargs):
            result = None
            start_time = time.time()
            logger.info(
                log_messages.get("start", "Starting LLM analysis."),
                extra={
                    "function": func.__name__,
                    "input_kwargs": {
                        "max_retries": max_retries,
                        "start_time": start_time,
                        "time_limit": time_limit,
                        "llm": llm,
                        "llm_messages": llm_messages,
                        "llm_kwargs": llm_kwargs,
                        "kwargs": kwargs,
                    },
                },
            )
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
                logger.error(
                    f"Result for {func.__name__} is None",
                    extra={
                        "function": func.__name__,
                        "input_kwargs": kwargs,
                        "response": result,
                    },
                )
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
    llm_kwargs: dict = None,
    **kwargs,
):
    result = None
    llm_kwargs = {} if llm_kwargs is None else llm_kwargs
    max_retries = 1 if max_retries is None else max_retries
    time_limit = 0 if time_limit is None else time_limit
    for i in range(max_retries):
        llm_response = llm.run(messages=llm_messages, **llm_kwargs)
        if llm_response is None:
            continue
        try:
            if "llm_response" in kwargs:
                del kwargs["llm_response"]
            result = func(llm_response=llm_response.message.content, **kwargs)
            if result is None:
                logger.error(
                    f"Result is None in iteration number {i + 1} of function {func.__name__}."
                )
                continue
            logger.info(
                f"Result from {func.__name__} recieved: {result}",
                extra={
                    "function": func.__name__,
                    "input_kwargs": {
                        "llm_response": llm_response.message.content,
                        **kwargs,
                    },
                    "response": {
                        "llm_response": llm_response.message.content,
                        "result": result,
                    },
                },
            )
            break
        except Exception as e:
            logger.error(
                f"Error in iteration number {i + 1} of function {func.__name__}. {e.__class__.__name__}: {e}.",
                extra={
                    "function": func.__name__,
                    "input_kwargs": {
                        "llm_response": llm_response.message.content,
                        **kwargs,
                    },
                    "response": result,
                    "traceback": traceback.format_exc().splitlines(),
                },
            )
            llm_messages.append(AssistantMessage(content=llm_response.message.content))
            llm_messages.append(
                SystemMessage(
                    content=f"Your response resulted in the following error:\n{traceback.format_exc()}\n\n"
                    "Read and understand the error CAREFULLY.\nUNDERSTAND how to FIX the error.\nCHANGE your code accordingly."
                )
            )
        finally:
            if (
                result is None
                and time.time() - start_time > time_limit
                and time_limit > 0
            ):
                logger.error(
                    f"Timeout in iteration number {i + 1} of function {func.__name__} with time limit of {time_limit} seconds.",
                    extra={
                        "function": func.__name__,
                        "input_kwargs": {
                            "llm_response": llm_response.message.content,
                            **kwargs,
                        },
                        "response": result,
                    },
                )
                raise AnalysisFailedError(
                    f"Timeout in iteration number {i + 1} of function {func.__name__} with time limit of {time_limit} seconds."
                )
    return result
