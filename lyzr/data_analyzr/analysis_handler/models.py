from typing import Annotated, Union, Any, Literal
from pydantic import BaseModel, Field, model_validator


class ConversionArgs(BaseModel):
    task: Literal[
        "convert_to_datetime",
        "convert_to_numeric",
        "convert_to_categorical",
        "convert_to_bool",
        "standard_scaler",
        "remove_nulls",
    ]
    columns: list[str] = Field(..., description="List of column names to be used.")


class TimePeriodArgs(BaseModel):
    task: Literal["extract_time_features"]
    time_col: str = Field(..., description="Column to extract the time period from.")
    feature_to_extract: Literal[
        "week",
        "month",
        "year",
        "day",
        "hour",
        "minute",
        "second",
        "weekday",
    ] = Field(
        ...,
        description="Time feature to extract from the datetime column.",
    )


class SelectValuesArgs(BaseModel):
    task: Literal["select_values"]
    columns: list[str] = Field(
        default=None, description="List of column names to select values from."
    )
    indices: list[int] = Field(
        default=None, description="List of indices to select from the columns."
    )


class MathOperationArgs(BaseModel):
    task: Literal[
        "add",
        "subtract",
        "multiply",
        "divide",
    ]
    columns: list[str] = Field(
        ..., description="List of column names to perform the operation on."
    )
    result: str = Field(..., description="Name of the column to store the result in.")


class RowStatsArgs(BaseModel):
    task: Literal[
        "row_wise_mean",
        "row_wise_median",
        "row_wise_mode",
        "row_wise_standard_deviation",
        "row_wise_sum",
        "row_wise_cumsum",
        "row_wise_cumprod",
    ]
    columns: list[str] = Field(
        ..., description="List of column names to perform the operation on."
    )
    result: str = Field(..., description="Name of the column to store the result in.")


class ColumnStatsArgs(BaseModel):
    task: Literal[
        "column_wise_mean",
        "column_wise_median",
        "column_wise_mode",
        "column_wise_standard_deviation",
        "column_wise_sum",
        "column_wise_cumsum",
        "column_wise_cumprod",
    ]
    column: str = Field(
        ..., description="Name of column on which to perform the operation."
    )
    result: str = Field(..., description="Name of the column to store the result in.")


class GroupbyArgs(BaseModel):
    task: Literal["groupby"]
    columns: list[str] = Field(..., description="List of column names to group by.")
    agg: Union[str, list] = Field(..., description="Aggregate function to apply.")
    agg_columns: list[str] = Field(
        ..., description="Column names to apply the aggregate function on."
    )


class SortValuesArgs(BaseModel):
    task: Literal["sortvalues"]
    columns: list[str] = Field(..., description="List of column names to sort.")
    ascending: bool = Field(
        default=True, description="Whether to sort in ascending order."
    )


class FilterArgs(BaseModel):
    task: Literal["filter"]
    columns: list[str] = Field(..., description="List of column names to filter.")
    values: list[Any] = Field(..., description="Values to compare the columns to.")
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
    ] = Field(
        ...,
        description="Relations to use for comparison.",
    )


class CorrelationArgs(BaseModel):
    task: Literal["correlation"]
    columns: list[str] = Field(
        ..., description="List of column names to calculate correlation."
    )
    method: Literal[
        "pearson",
        "kendall",
        "spearman",
    ] = Field(
        default="pearson",
        description="Method to calculate correlation.",
    )


class MLArgs(BaseModel):
    task: Literal["regression", "classification", "clustering"]
    x_columns: list[str] = Field(..., description="List of column names for x-axis.")
    y_columns: list[str] = Field(..., description="List of column names for y-axis.")


class ForecastArgs(BaseModel):
    task: Literal["forecast"]
    time_column: str = Field(..., description="Column name for time.")
    y_column: str = Field(..., description="Column name for y-axis.")
    end: str = Field(None, description="End date for forecasting.")
    steps: int = Field(None, description="Number of steps to forecast.")

    @model_validator(mode="before")
    def validate_end_or_steps(cls, values):
        if isinstance(values, dict):
            end = values.get("end")
            steps = values.get("steps")
        else:
            end = getattr(values, "end", None)
            steps = getattr(values, "steps", None)
        if (end is None) and (steps is None):
            raise ValueError("Either 'end' or 'steps' must be provided.")
        return values


ValidArgs = Annotated[
    Union[
        ConversionArgs,
        TimePeriodArgs,
        SelectValuesArgs,
        MathOperationArgs,
        ColumnStatsArgs,
        GroupbyArgs,
        SortValuesArgs,
        FilterArgs,
        CorrelationArgs,
        MLArgs,
        ForecastArgs,
    ],
    Field(..., description="Arguments for the task.", discriminator="task"),
]


class PythonicAnalysisStepDetails(BaseModel):
    step: int = Field(..., description="Step number starting from 1.")
    args: ValidArgs = Field(
        ...,
        description="Arguments for the task.",
        examples=[
            {"task": "standard_scaler", "columns": ["col1"]},
            {"task": "add", "columns": ["col1", "col2"], "result": "col3"},
            {
                "task": "groupby",
                "columns": ["col1"],
                "agg": "mean",
                "agg_col": ["col2"],
            },
        ],
    )


class PythonicAnalysisModel(BaseModel):
    analysis_df: str = Field(..., description="Name of the dataframe to be analyzed.")
    steps: list[PythonicAnalysisStepDetails] = Field(
        ..., description="List of preprocessing steps to be applied.", min_length=1
    )
    output_columns: list[str] = Field(
        ...,
        description="List of columns to be outputted after the analysis.",
    )
