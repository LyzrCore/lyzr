"""
Utility functions for handling analysis outputs, extracting information from LLM responses, and processing plot paths.
"""

# standard library imports
import re
import os
import string
import logging
from pathlib import Path
from typing import Any, Sequence, Union

# third-party imports
import numpy as np
import pandas as pd

# local imports
from lyzr.data_analyzr.utils import deterministic_uuid


def extract_python_code(llm_response: str) -> str:
    """
    Extracts Python code from a given string response.

    This function searches for Python code blocks within a string that follows the Markdown code block format.
    - It first attempts to find code blocks explicitly marked as Python (```python ... ```).
    - If no such block is found, it then searches for any generic code block (``` ... ```).
    - If no code blocks are found, the function returns the original string.

    Args:
        llm_response (str): The string response potentially containing Python code blocks.

    Returns:
        str: The extracted Python code if found, otherwise the original string.
    """
    py_code = re.search(r"```python\n(.*?)```", llm_response, re.DOTALL)
    if py_code:
        return py_code.group(1)
    py_code = re.search(r"```(.*?)```", llm_response, re.DOTALL)
    if py_code:
        return py_code.group(1)
    return llm_response


def extract_df_names(code: str, df_names: list[str]) -> list[str]:
    """
    Extracts and returns a list of DataFrame names that are present in the given code.

    Args:
        code (str): The code as a string in which to search for DataFrame names.
        df_names (list[str]): A list of DataFrame names to search for in the code.

    Returns:
        list[str]: A list of DataFrame names that are found in the code.
    """
    extracted_names = []
    for name in df_names:
        if name in code:
            extracted_names.append(name)
    return extracted_names


def remove_punctuation_from_string(value: str) -> str:
    """
    Remove punctuation from a given string, strip whitespace, and convert to lowercase.
    Used for extracting column names from code snippets.

    Args:
        value (str): The input string from which punctuation will be removed.

    Returns:
        str: The processed string with punctuation removed, whitespace stripped, and converted to lowercase.
    """
    value = str(value).strip()
    value = value.translate(str.maketrans("", "", string.punctuation))
    value = value.replace(" ", "").lower()
    return value


def extract_column_names(code: str, df_columns: list[str]) -> list[str]:
    """
    Extract column names from a given code string based on a list of DataFrame column names.

    Args:
        code (str): The code string from which to extract potential column names.
        df_columns (list[str]): A list of column names from a DataFrame to validate against.

    Returns:
        list[str]: A list of valid column names found within the code string.

    Procedure:
        - Search for double-quoted and single-quoted substrings in the code.
        - Remove punctuation from these substrings and compare them against the DataFrame column names.
        - If a match is found, add the original column name to the list of valid column names.
        - Return the list of valid column names found in the code.
    """
    cols_dict = {remove_punctuation_from_string(col): col for col in df_columns}
    regex = [r"\"(.*?)\"", r"'(.*?)'"]
    cols = []
    for reg in regex:
        for x in re.finditer(reg, code):
            for a in x.groups():
                if remove_punctuation_from_string(a).strip() != "":
                    cols.append(remove_punctuation_from_string(a))
    return list(set(cols_dict[c] for c in cols if c in cols_dict))


def remove_print_and_plt_show(code: str) -> str:
    """
    Removes lines containing:
        - print statements
        - plt.show()
        - plt.savefig()
        - fig.savefig()
    from the given code.

    Args:
        code (str): A string containing the code to be processed.

    Returns:
        str: The code with the specified lines removed.
    """
    codelines = code.split("\n")
    return "\n".join(
        [
            line
            for line in codelines
            if not (
                (line.strip().startswith("print("))
                or (line.strip().startswith("plt.show()"))
                or (line.strip().startswith("plt.savefig("))
                or (line.strip().startswith("fig.savefig("))
            )
        ]
    ).strip()


def handle_analysis_output(
    analysis_output: Any,
) -> Union[str, pd.DataFrame, dict[str, pd.DataFrame], None]:
    """
    Processes the given analysis output and returns it in a standardized format.

    Args:
        analysis_output (Any): The output from an analysis which can be of type
            None, pandas DataFrame, pandas Series, numpy number, int, float, str, bool, complex,
            Sequence, set, numpy ndarray, or dict.

    Returns:
        Union[str, pd.DataFrame, dict[str, pd.DataFrame], None]: The processed output in a standardized format:
            - None if the input is None.
            - pandas DataFrame if the input is a DataFrame or can be converted to one.
            - pandas DataFrame if the input is a Series.
            - string if the input is a numpy number, int, float, str, bool, complex, or a collection (Sequence, set, or 1D numpy ndarray).
            - pandas DataFrame if the input is a 2D numpy ndarray.
            - pandas DataFrame if the input is a dict that can be converted to a DataFrame.
            - string if the input is a dict that cannot be converted to a DataFrame.
            - string for any other type of input.
    """
    if analysis_output is None:
        return None
    if isinstance(analysis_output, pd.DataFrame):
        return analysis_output
    if isinstance(analysis_output, pd.Series):
        return analysis_output.to_frame()
    if isinstance(analysis_output, (np.number, int, float, str, bool, complex)):
        return str(analysis_output)
    if isinstance(analysis_output, (Sequence, set)):
        return ", ".join([str(i) for i in analysis_output])
    if isinstance(analysis_output, np.ndarray):
        if analysis_output.ndim == 1:
            return ", ".join([str(i) for i in analysis_output])
        return pd.DataFrame(analysis_output)
    if isinstance(analysis_output, dict):
        try:
            return pd.DataFrame(analysis_output)
        except ValueError:
            return handle_dict_output(analysis_output)[0]
    return str(analysis_output)


def handle_dict_output(
    analysis_output: dict,
) -> tuple[dict[str, Union[str, pd.DataFrame]], bool]:
    """
    Processes a dictionary of analysis outputs, converting values to strings or handling nested dictionaries.

    Args:
        analysis_output (dict): A dictionary where keys are strings and values can be of various types
            including nested dictionaries and pandas DataFrames.

    Returns:
        tuple: A tuple containing:
            - A dictionary with the same keys as the input, where values are either strings or pandas DataFrames.
            - A boolean indicating whether all values in the output dictionary are strings.
    """
    only_string_values = True
    output = {}
    for key, value in analysis_output.items():
        output_value = handle_analysis_output(value)
        if isinstance(output_value, pd.DataFrame):
            only_string_values = False
        elif isinstance(output_value, dict):
            output_value, output_string_values = handle_dict_output(output_value)
            only_string_values = output_string_values and only_string_values
        else:
            output_value = str(output_value)
        output[str(key)] = output_value
    if only_string_values:
        return (
            ", ".join([f"{k}: {v}" for k, v in analysis_output.items()]),
            only_string_values,
        )
    return output, only_string_values


def extract_sql(llm_response: str) -> str:
    """
    Extracts an SQL query from a given string response.

    This function searches for an SQL query in the provided response string.
    - It first attempts to find an SQL code block explicitly marked as SQL (```sql ... ```).
    - If not found, it searches for any content enclosed within triple backticks (``` ... ```).
    - If neither pattern is found, it returns the original response string.

    Args:
        llm_response (str): The response string potentially containing an SQL query.

    Returns:
        str: The extracted SQL query or the original response string if no query is found.
    """
    sql = re.search(r"```sql\n(.*?)```", llm_response, re.DOTALL)
    if sql:
        return sql.group(1)
    sql = re.search(r"```(.*?)```", llm_response, re.DOTALL)
    if sql:
        return sql.group(1)
    return llm_response


def handle_plotpath(
    plot_path: str, output_format: str, uuid: str, logger: logging.Logger
) -> str:
    """
    Handles the plot path for saving an image, ensuring the path is valid and writable.

    Args:
        plot_path (str): The initial path where the plot image is intended to be saved.
        output_format (str): The format of the output plot image (e.g., 'png', 'jpg').
        uuid (str): A unique identifier to be used in the random path if the provided path is incorrect.
        logger (logging.Logger): Logger instance for logging warnings and errors.

    Returns:
        str: A valid path where the plot image can be saved.

    Raises:
        Any exceptions encountered during the process are logged, and an alternative path is generated and returned.

    Logs:
        - Warning: If the provided plot path is incorrect, a warning is logged, and an alternative path is generated.

    Procedure:
        - If given uuid is None or an empty string, generate a new unique identifier.
        - Use _fix_plotpath to ensure the plot path is valid and has the correct output format.
        - Attempt to create an empty file at the fixed path.
        - If successful, return the fixed path.
        - If an exception is raised, log a warning, generate a random path, and return it instead.
    """
    if uuid is None or uuid.strip() == "":
        uuid = deterministic_uuid()
    uuid = uuid.strip()
    fixed_path = _fix_plotpath(plot_path, output_format, uuid)
    try:
        open(fixed_path, "w").close()
        return fixed_path
    except Exception:
        random_path = os.path.join("generated_plots", f"{uuid}.{output_format}")
        logger.warning(
            f"Incorrect path for plot image provided: {plot_path}. Using {random_path} instead.",
            extras={"function": "handle_plotpath"},
        )
        return handle_plotpath(random_path, output_format, uuid, logger)


def _fix_plotpath(plot_path: str, output_format: str, uuid: str) -> str:
    """
    Adjusts the given plot path to ensure it has the correct file extension and directory structure.

    Args:
        plot_path (str): The initial path for the plot, which can be a directory or a file path.
        output_format (str): The desired file extension for the plot (e.g., 'png', 'jpg').
        uuid (str): A unique identifier to be used in the random filename if the plot path is a directory.

    Returns:
        str: The adjusted plot path with the correct file extension and directory structure.

    Procedure:
        - If the plot path is None, generate a random filename with the specified output format.
        - Create parent directories in the path if they do not already exist.
        - If the plot path is a directory, generate and append a random filename with the specified output format.
        - Ensure the file extension of the plot path matches the output format.
    """
    if plot_path is None:
        plot_path = os.path.join("generated_plots", f"{uuid}.{output_format}")
    plot_path = Path(plot_path).as_posix()
    dir_path = os.path.dirname(plot_path)
    if dir_path.strip() != "":
        os.makedirs(dir_path, exist_ok=True)
    if os.path.isdir(plot_path):
        plot_path = os.path.join(plot_path, f"{uuid}.{output_format}")
    if os.path.splitext(plot_path)[1] != f".{output_format}":
        plot_path = os.path.splitext(plot_path)[0] + f".{output_format}"
    return plot_path


def make_locals_string(locals_: dict) -> str:
    """
    Generate a formatted string representation of a dictionary containing local variables.

    Args:
        locals_ (dict): A dictionary containing local variables to be formatted.

    Returns:
        str: A formatted string that represents the contents of the dictionary.
            - For pandas DataFrames, the first 5 rows are included in markdown format.
            - For nested dictionaries, the function recursively formats the contents.

    Procedure:
        - For each key-value pair in the dictionary:
            - If the value is a pandas DataFrame, include the first 5 rows in markdown format.
            - If the value is a nested dictionary, recursively format the contents.
            - Otherwise, include the key and value in the string.
        - Return the formatted string.
    """
    locals_str = "{\n"
    for name, value in locals_.items():
        if isinstance(value, pd.DataFrame):
            locals_str += f"{name} (DataFrame):\n{value.head(5).to_markdown()},\n"
        elif isinstance(value, dict):
            locals_str += f"{name} (dict): " + "{\n"
            for df_name, df in value.items():
                if isinstance(df, pd.DataFrame):
                    locals_str += (
                        f"{df_name} (DataFrame):\n{df.head(5).to_markdown()},\n"
                    )
                else:
                    locals_str += f"{df_name} ({type(df).__name__}): {df},\n"
            locals_str += "},\n"
        else:
            locals_str += f"{name} ({type(value).__name__}): {value},\n"
    return locals_str + "}"
