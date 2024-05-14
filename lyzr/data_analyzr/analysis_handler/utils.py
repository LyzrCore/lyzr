# standard library imports
import re
import logging
from typing import Any, Optional, Union, Literal

# third-party imports
import pandas as pd

# local imports
from lyzr.base.errors import DependencyError
from lyzr.data_analyzr.utils import get_columns_names

pd.options.mode.chained_assignment = None


class AnalysisExecutor:
    def __init__(self, df: pd.DataFrame, task: str, logger: logging.Logger):
        self.df = df
        self.logger = logger
        self.func = getattr(self, task.lower())

    def remove_nulls(self, columns: list) -> Union[pd.DataFrame, pd.Series]:
        columns = get_columns_names(df_columns=self.df.columns, columns=columns)
        self.df = self.df.dropna(subset=columns).reset_index(drop=True)

    def convert_to_datetime(self, columns: list):
        columns = get_columns_names(df_columns=self.df.columns, columns=columns)
        self.remove_nulls(columns=columns)
        self.df.loc[:, columns] = self.df.loc[:, columns].apply(
            pd.to_datetime, errors="coerce"
        )
        self.df = self.df.infer_objects()

    def convert_to_numeric(self, columns: list):
        columns = get_columns_names(df_columns=self.df.columns, columns=columns)
        self.remove_nulls(columns=columns)
        self.df = self._remove_punctuation(columns=columns, df=self.df)
        self.df.loc[:, columns] = self.df.loc[:, columns].apply(pd.to_numeric)
        self.df = self.df.infer_objects()

    def convert_to_categorical(self, columns: list):
        columns = get_columns_names(df_columns=self.df.columns, columns=columns)
        self.remove_nulls(columns=columns)
        self.df.loc[:, columns] = self.df.loc[:, columns].astype("category")
        self.df = self.df.infer_objects()

    def convert_to_bool(self, columns: list):
        columns = get_columns_names(df_columns=self.df.columns, columns=columns)
        self.remove_nulls(columns=columns)
        self.df.loc[:, columns] = self.df.loc[:, columns].astype("bool")
        self.df = self.df.infer_objects()

    def _remove_punctuation(
        self, columns: list, df: Union[pd.DataFrame, pd.Series]
    ) -> pd.DataFrame:
        if isinstance(df, pd.Series):
            df = df.to_frame()
        columns = get_columns_names(df_columns=self.df.columns, columns=columns)
        for col in columns:
            df.loc[:, col] = df.loc[:, col].apply(
                AnalysisExecutor._remove_punctuation_from_string
            )
        return df

    @staticmethod
    def _remove_punctuation_from_string(value):
        if not isinstance(value, str):
            return value
        value = value.strip()
        cleaned = re.sub(r"[^\d.]", "", str(value))
        if cleaned.replace(".", "").isdigit():
            return "-" + cleaned if value[0] == "-" else cleaned
        else:
            return value

    def one_hot_encode(self, columns: list):
        columns = get_columns_names(df_columns=self.df.columns, columns=columns)
        encoded_df = pd.get_dummies(self.df, columns=columns, dtype=float)
        categories = {}
        for col in columns:
            categories[col] = self.df[col].astype("category").cat.categories.to_list()
        self.df = encoded_df

    def ordinal_encode(self, columns: list):
        try:
            from sklearn.preprocessing import OrdinalEncoder  # type: ignore
        except ImportError:
            raise DependencyError({"scikit-learn": "scikit-learn==1.4.0"})

        columns = get_columns_names(df_columns=self.df.columns, columns=columns)
        encoder = OrdinalEncoder()
        encoded_df = self.df
        encoded_df.loc[:, columns] = encoder.fit_transform(
            self.df.loc[:, columns].values
        )
        categories = {}
        for col in columns:
            categories[col] = self.df[col].astype("category").cat.categories.to_list()
        self.df = encoded_df

    def standard_scaler(self, columns):
        try:
            from sklearn.preprocessing import StandardScaler  # type: ignore
        except ImportError:
            raise DependencyError({"scikit-learn": "scikit-learn==1.4.0"})

        columns = get_columns_names(df_columns=self.df.columns, columns=columns)
        scaler = StandardScaler()
        scaled_df = self.df
        scaled_df.loc[:, columns] = scaler.fit_transform(self.df.loc[:, columns].values)
        self.df = scaled_df

    def extract_time_features(
        self,
        time_col: str,
        feature_to_extract: Literal[
            "week", "month", "year", "day", "hour", "minute", "second", "weekday"
        ],
    ):
        time_col = get_columns_names(df_columns=self.df.columns, columns=[time_col])
        self.df.loc[:, time_col] = pd.to_datetime(self.df.loc[:, time_col])
        if feature_to_extract == "week":
            self.df.loc[:, ["week"]] = (
                pd.to_datetime(self.df.loc[:, time_col], errors="coerce")
                .dt.isocalendar()
                .week
            )
        elif feature_to_extract == "month":
            self.df.loc[:, ["month"]] = pd.to_datetime(
                self.df.loc[:, time_col], errors="coerce"
            ).dt.month
        elif feature_to_extract == "year":
            self.df.loc[:, ["year"]] = pd.to_datetime(
                self.df.loc[:, time_col], errors="coerce"
            ).dt.year
        elif feature_to_extract == "day":
            self.df.loc[:, ["day"]] = pd.to_datetime(
                self.df.loc[:, time_col], errors="coerce"
            ).dt.day
        elif feature_to_extract == "hour":
            self.df.loc[:, ["hour"]] = pd.to_datetime(
                self.df.loc[:, time_col], errors="coerce"
            ).dt.hour
        elif feature_to_extract == "minute":
            self.df.loc[:, ["minute"]] = pd.to_datetime(
                self.df.loc[:, time_col], errors="coerce"
            ).dt.minute
        elif feature_to_extract == "second":
            self.df.loc[:, ["second"]] = pd.to_datetime(
                self.df.loc[:, time_col], errors="coerce"
            ).dt.second
        elif feature_to_extract == "weekday":
            self.df.loc[:, ["weekday"]] = pd.to_datetime(
                self.df.loc[:, time_col], errors="coerce"
            ).dt.weekday

    def select_values(self, columns: list, indices: list):
        if indices is not None:
            self.df = self.df.loc[indices]
        if columns is not None:
            columns = get_columns_names(df_columns=self.df.columns, columns=columns)
            self.df = self.df.loc[:, columns]

    def add(self, columns: list, result: str):
        columns = get_columns_names(df_columns=self.df.columns, columns=columns)
        self.df.loc[:, result] = self.df.loc[:, columns].sum(axis=1)

    def subtract(self, columns: list, result: str):
        columns = get_columns_names(df_columns=self.df.columns, columns=columns)
        self.df.loc[:, result] = self.df[columns[0]] - self.df[columns[1]]

    def multiply(self, columns: list, result: str):
        columns = get_columns_names(df_columns=self.df.columns, columns=columns)
        self.df.loc[:, result] = self.df.loc[:, columns].prod(axis=1)

    def divide(self, columns: list, result: str):
        columns = get_columns_names(df_columns=self.df.columns, columns=columns)
        self.df = self.df[self.df[columns[1]] != 0]
        self.df.loc[:, result] = self.df[columns[0]] / self.df[columns[1]]

    def sortvalues(self, columns: list, ascending: Optional[Any] = True):
        if isinstance(ascending, list) and len(ascending) == len(columns):
            for col, asc in zip(columns, ascending):
                self.df = self._sorter([col], asc)
        elif isinstance(ascending, str) or isinstance(ascending, bool):
            self.df = self._sorter(columns, ascending)
        else:
            self.logger.warning(
                "Invalid value provided for ascending. Defaulting to True.",
                extra={"function": "AnalysisUtil.sortvalues"},
            )
            self.df = self._sorter(columns, True)

    def _sorter(self, columns: list, ascending: Any):
        if isinstance(ascending, str):
            if ascending.lower() == "true":
                ascending = True
            elif ascending.lower() == "false":
                ascending = False
            else:
                self.logger.warning(
                    f"Invalid value provided for ascending: {ascending}. Defaulting to True.",
                    extra={"function": "AnalysisUtil._sorter"},
                )
                ascending = True
        if not isinstance(ascending, bool):
            self.logger.warning(
                f"Invalid value type provided for ascending: {type(ascending)}. Defaulting to True.",
                extra={"function": "AnalysisUtil._sorter"},
            )
            ascending = True
        columns = get_columns_names(df_columns=self.df.columns, columns=columns)
        self.df = self.df.sort_values(columns, ascending=ascending)

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
    ):
        columns = get_columns_names(df_columns=self.df.columns, columns=columns)

        if len(columns) != len(values) and len(values) == 1:
            values = values * len(columns)
        if len(columns) != len(relations) and len(relations) == 1:
            relations = relations * len(columns)
        if len(columns) != len(values) or len(columns) != len(relations):
            self.logger.warning(
                "Invalid number of columns, values or relations provided. Returning original dataframe.",
                extra={"function": "AnalysisUtil.filter"},
            )
            return

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
                self.logger.warning(
                    f"Invalid relation provided: {rel}. Skipping.",
                    extra={"function": "AnalysisUtil.filter"},
                )

    def column_wise_mean(self, column: str, result: str):
        column = get_columns_names(df_columns=self.df.columns, columns=[column])[0]
        self.df.loc[:, result] = self.df.loc[:, column].mean(axis=0)

    def column_wise_median(self, column: str, result: str):
        column = get_columns_names(df_columns=self.df.columns, columns=[column])[0]
        self.df.loc[:, result] = self.df.loc[:, column].median(axis=0)

    def column_wise_mode(self, column: str, result: str):
        column = get_columns_names(df_columns=self.df.columns, columns=[column])[0]
        self.df.loc[:, result] = self.df.loc[:, column].mode(axis=0)

    def column_wise_standard_deviation(self, column: str, result: str):
        column = get_columns_names(df_columns=self.df.columns, columns=[column])[0]
        self.df.loc[:, result] = self.df.loc[:, column].std(axis=0)

    def column_wise_sum(self, column: str, result: str):
        column = get_columns_names(df_columns=self.df.columns, columns=[column])[0]
        self.df.loc[:, result] = self.df.loc[:, column].sum(axis=0)

    def column_wise_cumsum(self, column: str, result: str):
        column = get_columns_names(df_columns=self.df.columns, columns=[column])[0]
        self.df.loc[:, result] = self.df.loc[:, column].cumsum(axis=0)

    def column_wise_cumprod(self, column: str, result: str):
        column = get_columns_names(df_columns=self.df.columns, columns=[column])[0]
        self.df.loc[:, result] = self.df.loc[:, column].cumprod(axis=0)

    def row_wise_mean(self, columns: list, result: str):
        columns = get_columns_names(df_columns=self.df.columns, columns=columns)
        self.df.loc[:, result] = self.df.loc[:, columns].mean(axis=1)

    def row_wise_median(self, columns: list, result: str):
        columns = get_columns_names(df_columns=self.df.columns, columns=columns)
        self.df.loc[:, result] = self.df.loc[:, columns].median(axis=1)

    def row_wise_mode(self, columns: list, result: str):
        columns = get_columns_names(df_columns=self.df.columns, columns=columns)
        self.df.loc[:, result] = self.df.loc[:, columns].mode(axis=1)

    def row_wise_standard_deviation(self, columns: list, result: str):
        columns = get_columns_names(df_columns=self.df.columns, columns=columns)
        self.df.loc[:, result] = self.df.loc[:, columns].std(axis=1)

    def row_wise_sum(self, columns: list, result: str):  # NOSONAR
        columns = get_columns_names(df_columns=self.df.columns, columns=columns)
        self.df.loc[:, result] = self.df.loc[:, columns].sum(axis=1)

    def row_wise_cumsum(self, columns: list, result: str):
        columns = get_columns_names(df_columns=self.df.columns, columns=columns)
        self.df.loc[:, result] = self.df.loc[:, columns].cumsum(axis=1)

    def row_wise_cumprod(self, columns: list, result: str):
        columns = get_columns_names(df_columns=self.df.columns, columns=columns)
        self.df.loc[:, result] = self.df.loc[:, columns].cumprod(axis=1)

    def groupby(
        self,
        columns: list,
        agg: Union[str, list],
        agg_columns: Optional[list] = None,
    ):
        if (columns is None and agg_columns is None) or (
            len(columns) == 0 and len(agg_columns) == 0
        ):
            self.logger.warning(
                "No valid columns provided. Returning original dataframe.",
                extra={"function": "AnalysisUtil.groupby"},
            )
            return
        if (columns is None) or (len(columns) == 0):
            columns = agg_columns
        elif (agg_columns is None) or (len(agg_columns) == 0):
            agg_columns = columns
        columns = get_columns_names(df_columns=self.df.columns, columns=columns)
        agg_columns = get_columns_names(df_columns=self.df.columns, columns=agg_columns)
        self.df = self.df.groupby(columns)[agg_columns].agg(agg).reset_index()

    def correlation(
        self,
        columns: list,
        method: Optional[str] = "pearson",
        result: Optional[str] = None,
    ):
        columns = get_columns_names(df_columns=self.df.columns, columns=columns)
        if result is not None:
            self.df.loc[:, [result]] = self.df.loc[:, columns].corr(method=method)
        else:
            self.df = self.df.loc[:, columns].corr(method=method)

    def regression(self, x: list, y: list):
        try:
            from sklearn.linear_model import LinearRegression  # type: ignore
        except ImportError:
            raise DependencyError({"scikit-learn": "scikit-learn==1.4.0"})

        x = get_columns_names(df_columns=self.df.columns, columns=x)
        y = get_columns_names(df_columns=self.df.columns, columns=y)
        model = LinearRegression()
        self.df = model.fit(self.df.loc[:, x], self.df.loc[:, y])

    def forecast(
        self,
        time_column: str,
        y_column: str,
        end: Optional[str] = None,
        steps: Optional[int] = None,
    ):
        try:
            import pmdarima as pm  # type: ignore
        except ImportError:
            raise DependencyError({"pmdarima": "pmdarima==2.0.4"})

        time_column = get_columns_names(
            df_columns=self.df.columns, columns=[time_column]
        )[0]
        y_column = get_columns_names(df_columns=self.df.columns, columns=[y_column])[0]
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
        self.df = data
