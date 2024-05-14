from enum import Enum
from typing import Literal, Union
from pydantic import BaseModel, Field, model_validator


class ColumnDataType(str, Enum):
    datetime = "datetime"
    numeric = "numeric"
    string = "string"
    categorical = "categorical"
    bool = "bool"
    object = "object"

    @classmethod
    def values(cls):
        return [item.value for item in cls.__members__.values()]


class PlotingStepsDetails(BaseModel):
    subplot: tuple[int, int] = Field(
        default=(1, 1),
        description="The row and column number of the subplot respectively.",
    )
    plot_type: Literal["line", "bar", "barh", "scatter", "hist"] = Field(
        default=Literal["line"],
        description="The type of plot to be made, for example 'line', 'bar', 'barh', 'scatter', 'hist'.",
    )
    x: Union[str, None] = Field(
        default=None, description="Name of the column to be plot on the x-axis."
    )
    x_dtype: Union[ColumnDataType, None] = Field(
        default=None,
        description="Data type of the x-axis column.",
    )
    y: Union[str, None] = Field(
        default=None, description="Name of the column to be plot on the y-axis."
    )
    y_dtype: Union[ColumnDataType, None] = Field(
        default=None,
        description="Data type of the y-axis column.",
        examples=ColumnDataType.values(),
    )
    args: Union[dict, None] = Field(
        default_factory=dict,
        description="The arguments required to make the plot.",
        examples=[
            {"xlabel": "X-axis label", "ylabel": "Y-axis label"},
            {"color": "red", "linestyle": "--"},
            {"stacked": True},
            {"log": True},
        ],
    )

    @model_validator(mode="before")
    def set_stacked_value(cls, values):
        if not isinstance(values, dict):
            values = values.model_dump()
        if "bar" in values["plot_type"]:
            stacked = values["args"].get("stacked", False)
            if isinstance(stacked, str):
                if stacked.lower() == "true":
                    stacked = True
                elif stacked.lower() == "false":
                    stacked = False
                else:
                    stacked = False
            if not isinstance(stacked, bool):
                print(
                    f"Invalid value type provided for stacked: {type(values['args']['stacked'])}. Defaulting to False."
                )
                stacked = False
            values["args"]["stacked"] = stacked
        return values


class PlottingStepsModel(BaseModel):
    df_name: str = Field(..., description="Name of the dataframe to be plotted.")
    figsize: tuple[int, int] = Field(
        default=(15, 8),
        description="The width and height of the figure respectively.",
        examples=[(10, 8), (15, 15)],
    )
    subplots: tuple[int, int] = Field(
        default=(1, 1),
        description="The number of rows and columns of the subplot grid respectively.",
        examples=[(1, 2), (3, 1)],
    )
    title: str = Field(..., description="Title of the plot.")
    plots: list[PlotingStepsDetails] = Field(
        ..., description="List of plots to be plotted.", min_length=1
    )
