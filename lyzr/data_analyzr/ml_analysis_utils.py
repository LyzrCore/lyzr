# standard library imports
import re
import time
import logging
import traceback
from typing import Union, Optional, Literal, Any

# third-party imports
import pandas as pd

# local imports
from lyzr.base.base import SystemMessage, AssistantMessage
from lyzr.base.prompt import LyzrPromptFactory
from lyzr.base.llm import LiteLLM
from lyzr.data_analyzr.output_handler import (
    check_output_format,
    validate_output_step_details,
)
from lyzr.base.errors import DependencyError
from lyzr.data_analyzr.utils import get_columns_names


def run_analysis_step(
    data: pd.DataFrame,
    step_details: dict,
    logger: logging.Logger,
):
    logger.info(f"\nRunning step: {step_details}")
    if step_details["task"] == "clean_data":
        step_details["args"]["columns"] = get_columns_names(
            df_columns=data.columns,
            arguments=step_details["args"],
            logger=logger,
        )
        cleaner = CleanerUtil(
            data,
            step_details["type"],
            step_details["args"]["columns"],
        )
        data = cleaner.func()
    elif step_details["task"] == "transform":
        step_details["args"]["columns"] = get_columns_names(
            df_columns=data.columns,
            arguments=step_details["args"],
            logger=logger,
        )
        transformer = TransformerUtil(
            data,
            step_details["type"],
        )
        data = transformer.func(**step_details["args"])
    elif step_details["task"] == "math_operation":
        if "result" not in step_details["args"]:
            if len(step_details["args"].keys()) > 1:
                step_details["args"]["result"] = [
                    k for k in step_details["args"].keys() if k != "columns"
                ][0]
            else:
                step_details["args"]["result"] = "result"
        step_details["args"]["columns"] = get_columns_names(
            df_columns=data.columns,
            arguments=step_details["args"],
            logger=logger,
        )
        if len(step_details["args"]["columns"]) < 2:
            logger.info("Math operation requires at least 2 columns. Skipping.")
            return data
        operator = MathOperatorUtil(
            data,
            step_details["type"],
            logger,
            step_details["args"]["columns"],
            step_details["args"]["result"],
        )
        data = operator.func()
    elif step_details["task"] == "analysis":
        analyser = AnalyserUtil(data, step_details["type"], logger)
        data = analyser.func(**step_details["args"])
    return data


def print_df_details(df_dict: dict[pd.DataFrame], df_info_dict: dict[str]) -> str:
    str_output = ""
    for name, df in df_dict.items():
        var_name = name.lower().replace(" ", "_")
        if name in df_info_dict and isinstance(df, pd.DataFrame):
            str_output += f"Dataframe: `{var_name}`\nOutput of `{var_name}.head()`:\n{df.head()}\nOutput of `{var_name}.info()`:\n{df_info_dict[name]}\n"
        else:
            str_output += f"{name}: {df}\n"
    return str_output


class MLAnalysisFactory:
    def __init__(
        self,
        model: LiteLLM,
        data_dict: list[pd.DataFrame],
        data_info_dict: list[str],
        logger: logging.Logger,
        context: str,
    ):
        self.model = model
        self.model.set_model_kwargs(
            model_kwargs={"seed": 123, "temperature": 0.1, "top_p": 0.5}
        )
        self.df_dict = data_dict
        self.df_info_dict = data_info_dict
        self.context = context.strip() + "\n\n" if context != "" else ""
        self.logger = logger

    def _get_analysis_guide(self, user_input: str) -> str:
        output = self.model.run(
            messages=[
                LyzrPromptFactory(
                    name="ml_analysis_guide", prompt_type="system"
                ).get_message(
                    context=self.context,
                ),
                LyzrPromptFactory(
                    name="ml_analysis_guide", prompt_type="user"
                ).get_message(
                    df_details=print_df_details(self.df_dict, self.df_info_dict),
                    question=user_input,
                ),
            ],
            max_tokens=250,
        )
        return output.message.content.strip()

    def _get_analysis_steps_messages_kwargs(self, user_input: str) -> tuple:
        schema = {
            "analysis_df": str,  # Name of the dataframe
            "steps": [
                {
                    "step": 1,
                    "task": "clean_data",
                    "type": "convert_to_datetime",
                    "args": {
                        "columns": ["col1"],
                    },
                },
                {
                    "step": 2,
                    "task": "math_operation",
                    "type": "subtract",
                    "args": {
                        "columns": ["col1", "col2"],
                        "result": "col3",
                    },
                },
                {
                    "step": 3,
                    "task": "analysis",
                    "type": "groupby",
                    "args": {
                        "columns": ["col1"],
                        "agg": "mean",
                        "agg_col": ["col2"],
                    },
                },
            ],
            "output columns": ["col1", "col2", "col3"],
        }
        messages = [
            LyzrPromptFactory(
                name="ml_analysis_steps", prompt_type="system"
            ).get_message(schema=schema),
            LyzrPromptFactory(name="ml_analysis_steps", prompt_type="user").get_message(
                df_details=print_df_details(self.df_dict, self.df_info_dict),
                question=user_input,
                context=self.analysis_guide,
            ),
        ]
        return messages, dict(
            response_format={"type": "json_object"},
            max_tokens=2000,
            top_p=1,
        )
        # output = self.model.run(
        #     messages=[
        #         LyzrPromptFactory(
        #             name="analysis_steps", prompt_type="system"
        #         ).get_message(schema=schema),
        #         LyzrPromptFactory(
        #             name="analysis_steps", prompt_type="user"
        #         ).get_message(
        #             df_details=print_df_details(self.df_dict, self.df_info_dict),
        #             question=user_input,
        #             context=self.analysis_guide,
        #         ),
        #     ],
        #     response_format={"type": "json_object"},
        #     max_tokens=2000,
        #     top_p=1,
        # )
        # return output.message.content

    def _get_and_run_analysis(self, user_input) -> tuple:
        messages, kwargs = self._get_analysis_steps_messages_kwargs(user_input)
        for _ in range(5):
            try:
                llm_output = self.model.run(messages=messages, **kwargs).message.content
                messages.append(AssistantMessage(content=llm_output))
                self.logger.info("\nSecond analysis LLM output:\n{}".format(llm_output))
                self.analysis_dict = check_output_format(llm_output, self.logger)
                outputs, data = self.get_analysis_from_steps(self.analysis_dict)
                return outputs, data
            except RecursionError:
                raise RecursionError(
                    "The request could not be completed. Please wait a while and try again."
                )
            except Exception as e:
                if time.time() - self.start_time > 30:
                    raise TimeoutError(
                        "The request could not be completed. Please wait a while and try again."
                    )
                self.logger.info(f"{e.__class__.__name__}: {e}\n")
                self.logger.info("Traceback:\n{}\n".format(traceback.format_exc()))
                messages.append(
                    SystemMessage(
                        content=f"Your response resulted in the following error:\n{e.__class__.__name__}: {e}\n{traceback.format_exc()}\n\nPlease correct your response to prevent this error."
                    )
                )

    def get_analysis_from_steps(self, analysis_dict):
        df_name = analysis_dict["analysis_df"]
        data = self.df_dict[df_name].copy(deep=True)
        outputs = []
        for step_details in analysis_dict["steps"]:
            step_details = validate_output_step_details(step_details, self.logger)
            data = run_analysis_step(data, step_details, self.logger)
            if not isinstance(data, pd.DataFrame):
                outputs.append(data)
                data = self.df_dict[df_name].copy(deep=True)
        return outputs, data

    def _handle_single_output(self, data):
        self.logger.info(f"\nOutput dataframe shape: {data.shape}")
        self.logger.info(f"Output dataframe head:\n{data.head()}\n")

        if not isinstance(data, pd.DataFrame):
            self.output = data
            return

        output_columns = get_columns_names(
            df_columns=data.columns,
            columns=self.analysis_dict["output_columns"],
            logger=self.logger,
        )
        # Return output columns
        if data.size == 1:
            self.output = data[output_columns].values[0]
        else:
            self.output = data[output_columns]

    def _handle_list_output(self, outputs):
        self.logger.info(f"Output list length: {len(outputs)}\n")
        # self.logger.info(f"Output dataframe head: {outputs[0].head()}")

        self.output = outputs

    def run_complete_analysis(self, user_input: str) -> Union[list, pd.DataFrame]:
        # Get analysis steps from LLM using user_input
        self.analysis_guide = self._get_analysis_guide(user_input)
        if (
            self.analysis_guide
            == "The provided dataset is not sufficient to answer the question."
        ):
            return self.analysis_guide
        self.logger.info("First analysis LLM output:\n{}\n".format(self.analysis_guide))

        # Perform analysis using classes from lyzr/data_analyzr/run_analysis.py
        self.start_time = time.time()
        outputs, data = self._get_and_run_analysis(user_input)
        if len(outputs) > 0:
            outputs.append(data)
            self._handle_list_output(outputs)
        else:
            self._handle_single_output(data)

        return self.output


class CleanerUtil:

    def __init__(
        self,
        df: pd.DataFrame,
        cleaning_type: Literal[
            "remove_nulls",
            "convert_to_datetime",
            "convert_to_numeric",
            "convert_to_categorical",
        ],
        columns: list,
    ):
        self.df = df
        self.columns = columns
        self.func = getattr(self, cleaning_type.lower())

    def remove_nulls(self) -> pd.DataFrame:
        self.df.dropna(subset=self.columns, inplace=True)
        return self.df

    def convert_to_datetime(self) -> pd.DataFrame:
        self.df = self.remove_nulls()
        self.df.loc[:, self.columns] = self.df.loc[:, self.columns].apply(
            pd.to_datetime, errors="coerce"
        )
        return self.df.infer_objects()

    def convert_to_numeric(self) -> pd.DataFrame:
        self.df = self.remove_nulls()
        self.df = self._remove_punctuation()
        self.df.loc[:, self.columns] = self.df.loc[:, self.columns].apply(pd.to_numeric)
        return self.df.infer_objects()

    def convert_to_categorical(self) -> pd.DataFrame:
        self.df = self.remove_nulls()
        self.df.loc[:, self.columns] = self.df.loc[:, self.columns].astype("category")
        return self.df.infer_objects()

    def _remove_punctuation(self) -> pd.DataFrame:
        if isinstance(self.df, pd.Series):
            self.df = self.df.to_frame()
        for col in self.columns:
            self.df.loc[:, col] = self.df.loc[:, col].apply(
                CleanerUtil._remove_punctuation_from_string
            )
        return self.df

    @staticmethod
    def _remove_punctuation_from_string(value) -> str:
        if not isinstance(value, str):
            return value
        value = value.strip()
        cleaned = re.sub(r"[^\d.]", "", str(value))
        if cleaned.replace(".", "").isdigit():
            return "-" + cleaned if value[0] == "-" else cleaned
        else:
            return value


class TransformerUtil:
    def __init__(
        self,
        df: pd.DataFrame,
        transform_type: Literal[
            "one_hot_encode",
            "ordinal_encode",
            "scale",
            "extract_time_period",
            "select_indices",
        ],
    ):
        self.df = df
        self.func = getattr(
            self, transform_type.replace("-", "").replace(" ", "_").lower()
        )

    def one_hot_encode(self, columns) -> pd.DataFrame:
        encoded_df = pd.get_dummies(self.df, columns=columns, dtype=float)
        categories = {}
        for col in columns:
            categories[col] = self.df[col].astype("category").cat.categories.to_list()
        return encoded_df

    def ordinal_encode(self, columns) -> pd.DataFrame:
        try:
            from sklearn.preprocessing import OrdinalEncoder
        except ImportError:
            raise DependencyError({"scikit-learn": "scikit-learn==1.4.0"})

        encoder = OrdinalEncoder()
        encoded_df = self.df
        encoded_df.loc[:, columns] = encoder.fit_transform(
            self.df.loc[:, columns].values
        )
        categories = {}
        for col in columns:
            categories[col] = self.df[col].astype("category").cat.categories.to_list()
        return encoded_df

    def scale(self, columns) -> pd.DataFrame:
        try:
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            raise DependencyError({"scikit-learn": "scikit-learn==1.4.0"})

        scaler = StandardScaler()
        scaled_df = self.df
        scaled_df.loc[:, columns] = scaler.fit_transform(self.df.loc[:, columns].values)
        return scaled_df

    def extract_time_period(
        self,
        columns,
        period_to_extract: Literal[
            "week", "month", "year", "day", "hour", "minute", "second", "weekday"
        ],
    ) -> pd.DataFrame:
        columns = columns[0]
        self.df.loc[:, columns] = pd.to_datetime(self.df.loc[:, columns])
        if period_to_extract == "week":
            self.df.loc[:, ["week"]] = (
                pd.to_datetime(self.df.loc[:, columns], errors="coerce")
                .dt.isocalendar()
                .week
            )
        elif period_to_extract == "month":
            self.df.loc[:, ["month"]] = pd.to_datetime(
                self.df.loc[:, columns], errors="coerce"
            ).dt.month
        elif period_to_extract == "year":
            self.df.loc[:, ["year"]] = pd.to_datetime(
                self.df.loc[:, columns], errors="coerce"
            ).dt.year
        elif period_to_extract == "day":
            self.df.loc[:, ["day"]] = pd.to_datetime(
                self.df.loc[:, columns], errors="coerce"
            ).dt.day
        elif period_to_extract == "hour":
            self.df.loc[:, ["hour"]] = pd.to_datetime(
                self.df.loc[:, columns], errors="coerce"
            ).dt.hour
        elif period_to_extract == "minute":
            self.df.loc[:, ["minute"]] = pd.to_datetime(
                self.df.loc[:, columns], errors="coerce"
            ).dt.minute
        elif period_to_extract == "second":
            self.df.loc[:, ["second"]] = pd.to_datetime(
                self.df.loc[:, columns], errors="coerce"
            ).dt.second
        elif period_to_extract == "weekday":
            self.df.loc[:, ["weekday"]] = pd.to_datetime(
                self.df.loc[:, columns], errors="coerce"
            ).dt.weekday
        return self.df

    def select_indices(self, columns, indices) -> pd.DataFrame:
        return self.df.loc[indices, :]


class MathOperatorUtil:
    def __init__(
        self,
        df: pd.DataFrame,
        operator_type: Literal["add", "subtract", "multiply", "divide"],
        logger: logging.Logger,
        columns: list,
        result: str,
    ):
        self.df = df
        self.logger = logger
        self.columns = columns
        self.result = result
        self.df = self.df.dropna(subset=self.columns)
        self.func = getattr(self, operator_type.lower())

    def add(self) -> pd.DataFrame:
        self.df.loc[:, self.result] = self.df.loc[:, self.columns].sum(axis=1)
        return self.df

    def subtract(self) -> pd.DataFrame:
        self.df.loc[:, self.result] = (
            self.df[self.columns[0]] - self.df[self.columns[1]]
        )
        return self.df

    def multiply(self) -> pd.DataFrame:
        self.df.loc[:, self.result] = self.df.loc[:, self.columns].prod(axis=1)
        return self.df

    def divide(self) -> pd.DataFrame:
        self.df = self.df[self.df[self.columns[1]] != 0]
        self.df.loc[:, self.result] = (
            self.df[self.columns[0]] / self.df[self.columns[1]]
        )
        return self.df


class AnalyserUtil:
    def __init__(
        self,
        df: pd.DataFrame,
        analysis_type: Literal[
            "sortvalues",
            "filter",
            "mean",
            "sum",
            "cumsum",
            "groupby",
            "correlation",
            "regression",
        ],
        logger: logging.Logger,
    ):
        self.df = df
        self.logger = logger
        self.func = getattr(self, analysis_type.lower())

    def sortvalues(
        self, columns: list, ascending: Optional[Any] = True
    ) -> pd.DataFrame:
        if isinstance(ascending, list) and len(ascending) == len(columns):
            for col, asc in zip(columns, ascending):
                self.df = self._sorter([col], asc)
        elif isinstance(ascending, str) or isinstance(ascending, bool):
            self.df = self._sorter(columns, ascending)
        else:
            self.logger.warning(
                "Invalid value provided for ascending. Defaulting to True."
            )
            self.df = self._sorter(columns, True)
        return self.df

    def _sorter(self, columns: list, ascending: Any) -> pd.DataFrame:
        if isinstance(ascending, str):
            if ascending.lower() == "true":
                ascending = True
            elif ascending.lower() == "false":
                ascending = False
            else:
                self.logger.warning(
                    f"Invalid value provided for ascending: {ascending}. Defaulting to True."
                )
                ascending = True
        if not isinstance(ascending, bool):
            self.logger.warning(
                f"Invalid value type provided for ascending: {type(ascending)}. Defaulting to True."
            )
            ascending = True
        columns = get_columns_names(
            df_columns=self.df.columns, columns=columns, logger=self.logger
        )
        return self.df.sort_values(columns, ascending=ascending)

    def filter(
        self,
        columns: list[str],
        values: list[Any],
        relations: list[
            Literal[
                "lessthan",
                "greaterthan",
                "lessthanorequalto",
                "greaterthanorequalto",
                "equalto",
                "notequalto",
                "startswith",
                "endswith",
                "contains",
            ]
        ],
    ) -> pd.DataFrame:
        columns = get_columns_names(
            df_columns=self.df.columns, columns=columns, logger=self.logger
        )

        if len(columns) != len(values) and len(values) == 1:
            values = values * len(columns)
        if len(columns) != len(relations) and len(relations) == 1:
            relations = relations * len(columns)
        if len(columns) != len(values) or len(columns) != len(relations):
            self.logger.warning(
                "Invalid number of columns, values or relations provided. Returning original dataframe."
            )
            return self.df

        for i in range(len(columns)):
            col, val, rel = columns[i], values[i], relations[i]
            if rel == "startswith":
                self.df = self.df[self.df[col].str.startswith(str(val))]
            elif rel == "endswith":
                self.df = self.df[self.df[col].str.endswith(str(val))]
            elif rel == "contains":
                self.df = self.df[self.df[col].str.contains(str(val))]
            elif rel == "lessthan":
                self.df = self.df[self.df[col] < val]
            elif rel == "greaterthan":
                self.df = self.df[self.df[col] > val]
            elif rel == "lessthanorequalto":
                self.df = self.df[self.df[col] <= val]
            elif rel == "greaterthanorequalto":
                self.df = self.df[self.df[col] >= val]
            elif rel == "equalto":
                self.df = self.df[self.df[col] == val]
            elif rel == "notequalto":
                self.df = self.df[self.df[col] != val]
            else:
                self.logger.warning(f"Invalid relation provided: {rel}. Skipping.")
        return self.df

    def mean(self, columns: list, result: Optional[str] = None) -> pd.DataFrame:
        columns = get_columns_names(
            df_columns=self.df.columns, columns=columns, logger=self.logger
        )
        if result is not None:
            self.df.loc[:, result] = self.df.loc[:, columns].mean()
            return self.df
        return self.df.loc[:, columns].mean()

    def sum(self, columns: list, result: Optional[str] = None) -> pd.DataFrame:
        columns = get_columns_names(
            df_columns=self.df.columns, columns=columns, logger=self.logger
        )
        if result is not None:
            self.df.loc[:, result] = self.df.loc[:, columns].sum(axis=1)
            return self.df
        return self.df.loc[:, columns].sum()

    def cumsum(self, columns: list, result: Optional[str] = None) -> pd.DataFrame:
        columns = get_columns_names(
            df_columns=self.df.columns, columns=columns, logger=self.logger
        )
        if result is not None:
            self.df.loc[:, result] = self.df.loc[:, columns].cumsum()
            return self.df
        return self.df.loc[:, columns].cumsum()

    def groupby(
        self,
        columns: list,
        agg: Union[str, list],
        result: Optional[str] = None,
        agg_col: Optional[list] = None,
    ) -> pd.DataFrame:
        columns = get_columns_names(
            df_columns=self.df.columns, columns=columns, logger=self.logger
        )
        if len(columns) == 0:
            self.logger.warning(
                "No valid columns provided. Returning original dataframe."
            )
            return self.df
        if agg_col is None:
            agg_col = self.df.columns.to_list()
        agg_col = get_columns_names(
            df_columns=self.df.columns, columns=agg_col, logger=self.logger
        )
        if result is not None:
            self.df.loc[:, result] = self.df.groupby(columns)[agg_col].agg(agg)
            return self.df
        return self.df.groupby(columns)[agg_col].agg(agg).reset_index()

    def correlation(
        self,
        columns: list,
        method: Optional[str] = "pearson",
        result: Optional[str] = None,
    ) -> pd.DataFrame:
        columns = get_columns_names(
            df_columns=self.df.columns, columns=columns, logger=self.logger
        )
        if result is not None:
            self.df.loc[:, result] = self.df.loc[:, columns].corr(method=method)
            return self.df
        return self.df.loc[:, columns].corr(method=method)

    def regression(self, x: list, y: list) -> pd.DataFrame:
        try:
            from sklearn.linear_model import LinearRegression
        except ImportError:
            raise DependencyError({"scikit-learn": "scikit-learn==1.4.0"})

        x = get_columns_names(df_columns=self.df.columns, columns=x, logger=self.logger)
        y = get_columns_names(df_columns=self.df.columns, columns=y, logger=self.logger)
        model = LinearRegression()
        model.fit(self.df.loc[:, x], self.df.loc[:, y])
        return model

    def forecast(
        self,
        time_column: str,
        y_column: str,
        end: Optional[str] = None,
        steps: Optional[int] = None,
    ):
        try:
            import pmdarima as pm
        except ImportError:
            raise DependencyError({"pmdarima": "pmdarima==2.0.4"})

        time_column = get_columns_names(
            df_columns=self.df.columns, columns=[time_column], logger=self.logger
        )[0]
        y_column = get_columns_names(
            df_columns=self.df.columns, columns=[y_column], logger=self.logger
        )[0]
        data = self.df.loc[
            self.df[time_column].drop_duplicates().sort_values().index,
            [time_column, y_column],
        ]
        data = data.set_index([time_column])
        data.index = pd.DatetimeIndex(pd.to_datetime(data.index))
        model = pm.auto_arima(data[y_column])
        if end is not None:
            steps = len(
                pd.date_range(
                    start=data.index[-1],
                    end=pd.to_datetime(end),
                    freq=data.index.inferred_freq,
                )
            )
        y_pred = model.predict(n_periods=steps)
        for datetime in y_pred.index:
            data.loc[datetime, y_column] = y_pred[datetime]
        return data
