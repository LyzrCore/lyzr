"""
Logging functions for the lyzr package.
"""

import os
import re
import sys
import logging
import logging.handlers


def set_logger(
    name: str = None,
    logfilename: str = None,
    log_level: str = "INFO",
    print_log: bool = False,
):
    """
    Sets up a logger with specified configurations.

    Args:
        name (str, optional): The name of the logger. Defaults to "lyzr".
        logfilename (str, optional): The filename for the log file. Defaults to "lyzr".
        log_level (str, optional): The logging level (e.g., "INFO", "DEBUG"). Defaults to "INFO".
        print_log (bool, optional): If True, logs will also be printed to the console. Defaults to False.

    Returns:
        logging.Logger: Configured logger instance.

    Raises:
        ValueError: If an invalid log level is provided.

    Notes:
        - The log file will be in CSV format.
        - The logger will use a rotating file handler with a maximum file size of 25MB and up to 25 backup files.
        - Environment variables "log_level" and "logfilepath" will be set to the specified log level and log file path, respectively.
    """
    name = "lyzr" if name is None else name
    logfilename = "lyzr" if logfilename is None else logfilename
    logfilename = os.path.relpath(os.path.splitext(logfilename)[0] + ".csv")
    if os.path.dirname(logfilename).strip() != "":
        os.makedirs(os.path.dirname(logfilename), exist_ok=True)
    logger = logging.getLogger(name=name)
    if logger.hasHandlers():
        for handler in logger.handlers:
            try:
                handler.close()
            except Exception:
                pass
        logger.handlers.clear()
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError("Invalid log level: %s" % log_level)
    logger.setLevel(numeric_level)
    # write to console
    if print_log:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                get_console_log_format(),
                datefmt="%Y-%m-%d %H:%M:%S %Z",
                style="{",
            )
        )
        handler.setLevel(log_level)
        logger.addHandler(handler)
    # write to rotating csv files
    filehandler = logging.handlers.RotatingFileHandler(
        filename=logfilename,
        mode="a",
        maxBytes=26214400,  # maxBytes=25MB
        backupCount=25,
    )
    fileformatter = CustomFormatter(
        get_csv_log_format(),
        datefmt="%Y-%m-%d %H:%M:%S %Z",
        style="{",
    )
    filehandler.setLevel(logging.INFO)
    filehandler.setFormatter(fileformatter)
    logger.addHandler(filehandler)

    os.environ["log_level"] = str(log_level)
    os.environ["logfilepath"] = logfilename
    return logger


class CustomFormatter(logging.Formatter):
    """
    Custom logging formatter to handle dynamic fields and format log records.

    Methods:
        format(record: logging.LogRecord) -> str
            Formats the specified log record as text. Ensures that all dynamic fields
            specified in the format string are present in the log record, setting them
            to None if they are missing. Handles special formatting for the 'traceback'
            field and ensures that newlines are properly represented.
    """

    def format(self, record: logging.LogRecord) -> str:
        arg_pattern = re.compile(r"\{(\w+)\}")
        arg_names = [x.group(1) for x in arg_pattern.finditer(self._fmt)]
        for field in arg_names:
            if not hasattr(record, field):
                setattr(record, field, None)
            record_value = getattr(record, field)
            if (
                field == "traceback"
                and isinstance(record_value, str)
                and (
                    (record_value.strip()[0] != "[")
                    and (record_value.strip()[-1] != "]")
                )
            ):
                setattr(record, field, record_value.splitlines())
            setattr(record, field, str(record_value).replace("\n", "\\n"))
        record.msg = record.getMessage().strip()
        record.msg = record.getMessage().replace("\n", "\\n")
        return super().format(record)


def get_csv_log_format():
    """Returns the format string for logging records to CSV files."""
    columns = [
        "asctime",
        "name",
        "levelname",
        "filename",
        "funcName",
        "lineno",
        "message",
        "function",
        "traceback",
        "input_args",
        "input_kwargs",
        "response",
    ]
    sep = ",;"
    log_format = sep.join(["{" + x + "}" for x in columns])
    return log_format


def get_console_log_format():
    """Returns the format string for logging records to the console."""
    log_format = "{asctime} | {name} — {levelname} — {filename}: {funcName}: {lineno} — {message}"
    return log_format


def read_csv_log(filename: str):
    """
    Reads a CSV log file and returns it as a pandas DataFrame.

    Args:
        filename (str): The path to the CSV log file.

    Returns:
        pandas.DataFrame: A DataFrame containing the log data with predefined columns.

    Notes:
        - The expected columns in the CSV file are:
            - asctime
            - name
            - levelname
            - filename
            - funcName
            - lineno
            - message
            - function
            - traceback
            - input_args
            - input_kwargs
            - response
        - The CSV file is expected to use ",;" as the separator.
    """
    import pandas as pd

    columns = [
        "asctime",
        "name",
        "levelname",
        "filename",
        "funcName",
        "lineno",
        "message",
        "function",
        "traceback",
        "input_args",
        "input_kwargs",
        "response",
    ]
    sep = ",;"
    df = pd.read_csv(filename, sep=sep, names=columns, engine="python")
    return df
