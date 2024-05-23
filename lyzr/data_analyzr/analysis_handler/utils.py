# standard library imports
import re
import os
import string
import logging
from typing import Any, Sequence, Union

# third-party imports
import numpy as np
import pandas as pd

# local imports
from lyzr.data_analyzr.utils import deterministic_uuid


def extract_python_code(llm_response: str) -> str:
    py_code = re.search(r"```python\n(.*?)```", llm_response, re.DOTALL)
    if py_code:
        return py_code.group(1)
    py_code = re.search(r"```(.*?)```", llm_response, re.DOTALL)
    if py_code:
        return py_code.group(1)
    return llm_response


def extract_df_names(code: str, df_names: list[str]) -> list[str]:
    extracted_names = []
    for name in df_names:
        if name in code:
            extracted_names.append(name)
    return extracted_names


def remove_punctuation_from_string(value: str) -> str:
    value = str(value).strip()
    value = value.translate(str.maketrans("", "", string.punctuation))
    value = value.replace(" ", "").lower()
    return value


def extract_column_names(code: str, df_columns: list[str]) -> list[str]:
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
    # If the llm_response contains a markdown code block, with or without the sql tag, extract the sql from it
    sql = re.search(r"```sql\n(.*?)```", llm_response, re.DOTALL)
    if sql:
        return sql.group(1)
    sql = re.search(r"```(.*?)```", llm_response, re.DOTALL)
    if sql:
        return sql.group(1)
    return llm_response


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


def make_locals_string(locals_: dict) -> str:
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
