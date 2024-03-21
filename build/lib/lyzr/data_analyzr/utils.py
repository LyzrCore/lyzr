# standart-library imports
import io
import re

# third-party imports
import numpy as np
import pandas as pd


def _remove_punctuation_from_string(value) -> str:
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
            df.loc[:, col] = df.loc[:, col].apply(_remove_punctuation_from_string)
            df.loc[:, col] = pd.to_numeric(df.loc[:, col])
            df.loc[:, col] = df.loc[:, col].astype("float")
        except Exception:
            pass
    return df


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


def _format_df_dict_head(df_dict: dict[pd.DataFrame]) -> str:
    return "\n".join([f"{name}:\n{df.head()}\n" for name, df in df_dict.items()])
