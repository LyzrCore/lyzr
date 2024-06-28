"""
Utility functions for the data analyzr module in Lyzr.
"""

# standart-library imports
import re
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


def deterministic_uuid(content: Union[str, bytes, list] = None):
    """
    Generate a deterministic UUID based on the provided content.

    Creates a deterministic UUID by hashing the provided content using MD5.
    The content can be a string, bytes, or a list. If no content is provided, the current time is used.

    Args:
        content (Union[str, bytes, list], optional):
            The content to be hashed. Can be a string, bytes, or a list. Defaults to None.

    Returns:
        str: The generated deterministic UUID as a hexadecimal string.
    """
    if content is None:
        content = str(time.time())
    if isinstance(content, list):
        content = "/".join([str(x) for x in content if x is not None])
    if isinstance(content, str):
        content = content.encode("utf-8")
    hash_object = hashlib.md5(content)
    return hash_object.hexdigest()


def translate_string_name(name: str) -> str:
    """
    Converts a given string into a standardized format.

    Args:
        name (str): The input string to be transformed.

    Returns:
        str: The transformed string.

    Procedure:
    - Converts all characters to lowercase.
    - Strips leading and trailing whitespace.
    - Replaces all punctuation and spaces with underscores.
    - Removes leading and trailing underscores.
    """
    punc = string.punctuation + " "
    new_name = (
        name.lower().strip().translate(str.maketrans(punc, "_" * len(punc))).strip("_")
    )
    return re.sub(r"_+", "_", new_name)


def format_analysis_output(output_df, name: str = None) -> str:
    """
    Formats the details of a DataFrame, Series, list of DataFrames, or dictionary of DataFrames into a string.

    Formats inputs them into a string representation, depending on the input type:
    - pandas Series: Converts the Series to a DataFrame and formats it.
    - list: Recursively formats each element in the list.
    - dictionary: Recursively formats each value in the dictionary.
    - Any other input: Converts the input to a string.
    - pandas DataFrame: Uses a helper function `df_details_with_describe` to format the DataFrame details.

    Args:
        output_df (Union[pd.DataFrame, pd.Series, list, dict, Any]): The input data to be formatted.
        name (str, optional): An optional name to be used in the formatting. Defaults to None.

    Returns:
        str: The formatted string representation of the input data.
    """
    if isinstance(output_df, pd.Series):
        output_df = output_df.to_frame()
    if isinstance(output_df, list):
        return "\n".join([format_analysis_output(df) for df in output_df])
    if isinstance(output_df, dict):
        return "\n".join(
            [format_analysis_output(df, name) for name, df in output_df.items()]
        )
    if not isinstance(output_df, pd.DataFrame):
        return str(output_df)
    else:
        return df_details_with_describe(output_df, name)


def df_details_with_describe(
    output_df: Union[None, pd.DataFrame], name: str = None
) -> str:
    """
    Provides a detailed string representation of a DataFrame, including a snapshot and descriptive statistics.

    Generates a string that includes:
    - The name of the DataFrame.
    - A snapshot of the DataFrame (first and last 50 rows if the DataFrame is large).
    - Descriptive statistics of the DataFrame columns if the DataFrame is large.
    - If input is None, the function returns a string indicating that the DataFrame is None.

    Args:
        output_df (Union[None, pd.DataFrame]): The DataFrame to be described.
        name (str, optional): The name of the DataFrame. Defaults to "Dataframe".

    Returns:
        str: A string representation of the DataFrame details.
    """
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
    """
    Converts a DataFrame to a formatted string representation.

    Args:
        output_df (pd.DataFrame): The DataFrame to be converted to a string.

    Returns:
        str: The formatted string representation of the DataFrame.

    Procedure:
    - Convert all column names to strings.
    - Identify and convert columns with date or time information to datetime objects.
    - Format the DataFrame to a string with specific formatting for floats and datetime columns.
    """
    output_df.columns = [str(col) for col in output_df.columns.tolist()]
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
    """
    Formats a pandas Timestamp object to a string, using the format "dd MMM YYYY HH:MM"

    Args:
        date (pd.Timestamp): The pandas Timestamp object to be formatted.

    Returns:
        str: The formatted string representation of the Timestamp object.
    """
    return date.strftime("%d %b %Y %H:%M")


def logging_decorator(logger: logging.Logger):
    """
    A decorator that logs the execution of a function using a specified logger.

    This decorator logs the start and completion of the function execution, including the
    input arguments and the result. If an exception occurs during the function execution,
    it logs the error along with the traceback.

    Args:
        logger (logging.Logger): The logger instance to be used for logging.

    Returns:
        function: The decorated function with added logging functionality.

    Example:
        import logging
        logger = logging.getLogger(__name__)
        @logging_decorator(logger)
        def example_function(x, y):
            return x + y
    """

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
