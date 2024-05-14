import os
import re
import sys
import logging
import logging.handlers

# pip install python-json-logger


def set_logger(
    name: str = None,
    logfilename: str = None,
    log_level: str = "INFO",
    print_log: bool = False,
):
    name = "data_analyzr" if name is None else name
    logfilename = "data-analyzr" if logfilename is None else logfilename
    logfilename = os.path.relpath(os.path.splitext(logfilename)[0] + ".csv")
    logfilename = (
        os.path.join("lyzr-logs", logfilename)
        if os.path.dirname(logfilename).strip() == ""
        else logfilename
    )
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

    def format(self, record: logging.LogRecord) -> str:
        arg_pattern = re.compile(r"\{(\w+)\}")
        arg_names = [x.group(1) for x in arg_pattern.finditer(self._fmt)]
        for field in arg_names:
            if field not in record.__dict__:
                record.__dict__[field] = None
            if field == "traceback" and isinstance(record.__dict__[field], str):
                record.__dict__[field] = record.__dict__[field].splitlines()
            if isinstance(record.__dict__[field], str):
                record.__dict__[field] = record.__dict__[field].replace("\n", "\\n")
        record.msg = record.getMessage().strip()
        record.msg = record.getMessage().replace("\n", "\\n")
        return super().format(record)


def get_csv_log_format():
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
    log_format = "{asctime} | {name} — {levelname} — {filename}: {funcName}: {lineno} — {message}"
    return log_format


def read_csv_log(filename: str):
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
