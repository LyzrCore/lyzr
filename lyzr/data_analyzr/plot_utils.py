# standard library imports
import io
import os
import logging
from typing import Literal, Union

# third-party imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# local imports
from lyzr.base.prompt import Prompt
from lyzr.base.llms import LLM, set_model_params
from lyzr.data_analyzr.output_handler import check_output_format


class Plot:
    def __init__(self, x: list, y: list):
        self.x = x
        self.y = y


def print_df_details(df_dict: dict[pd.DataFrame], df_info_dict: dict[str]) -> str:
    str_output = ""
    for name, df in df_dict.items():
        var_name = name.lower().replace(" ", "_")
        if name in df_info_dict and isinstance(df, pd.DataFrame):
            str_output += f"Dataframe: `{var_name}`\nOutput of `{var_name}.head()`:\n{df.head()}\nOutput of `{var_name}.info()`:\n{df_info_dict[name]}\n"
        else:
            str_output += f"{name}: {df}\n"
    return str_output


class PlotFactory:
    def __init__(
        self,
        plotting_model: LLM,
        plotting_model_kwargs: dict,
        df_dict: list[pd.DataFrame],
        logger: logging.Logger,
        plot_context: str,
        plot_path: str,
    ):
        self.model = plotting_model
        self.model_kwargs = plotting_model_kwargs or {}
        self.model_kwargs = set_model_params(
            {"seed": 123, "temperature": 0.1, "top_p": 0.5}, self.model_kwargs
        )

        self.df_dict = df_dict
        self.df_info_dict = {}
        for df_name in self.df_dict:
            buffer = io.StringIO()
            self.df_dict[df_name].info(buf=buffer)
            self.df_info_dict[df_name] = buffer.getvalue()

        self.logger = logger
        self.context = plot_context

        self.plotting_library = "matplotlib"
        self.output_format = "png"

        self.plot_path = plot_path or "plot.png"
        if not self.plot_path.endswith(".png"):
            self.plot_path += ".png"

    def _get_plotting_guide(self, user_input: str) -> str:
        self.model.set_messages(
            messages=[
                {
                    "role": "system",
                    "content": Prompt("analysis_guide_pt")
                    .format(
                        df_details=print_df_details(self.df_dict, self.df_info_dict),
                        question=user_input,
                        context=self.context,
                        plotting_lib=self.plotting_library,
                    )
                    .text,
                },
            ]
        )
        self.model_kwargs = set_model_params(
            {"max_tokens": 500},
            self.model_kwargs,
        )
        output = self.model.run(**self.model_kwargs)
        return output.choices[0].message.content

    def _get_plotting_steps(self, user_input: str) -> str:
        schema = {
            "preprocess": {
                "analysis_df": str,
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
            },
            "plot": {
                "figsize": (int, int),
                "subplots": (int, int),
                "title": str,
                "plots": [
                    {
                        "subplot": (int, int),
                        "plot_type": "line",  # line, bar, barh, scatter, hist
                        "x": str,
                        "y": str,
                        "args": {
                            "xlabel": str,
                            "ylabel": str,
                            "color": str,
                            "linestyle": str,  # for line plots
                        },
                    },
                    {
                        "subplot": (int, int),
                        "plot_type": "bar",  # line, bar, barh, scatter, hist
                        "x": str,
                        "y": str,
                        "args": {
                            "xlabel": str,
                            "ylabel": str,
                            "color": str,
                            "stacked": bool,  # for bar plots
                        },
                    },
                ],
            },
        }
        self.model.set_messages(
            messages=[
                {
                    "role": "system",
                    "content": Prompt("plotting_steps_pt")
                    .format(
                        plotting_lib=self.plotting_library,
                        guide=self.plotting_guide,
                        schema=schema,
                        question=user_input,
                        df_details=print_df_details(self.df_dict, self.df_info_dict),
                    )
                    .text,
                },
            ]
        )
        self.model_kwargs = set_model_params(
            {
                "response_format": {"type": "json_object"},
                "max_tokens": 2000,
                "top_p": 1,
            },
            self.model_kwargs,
        )
        output = self.model.run(**self.model_kwargs)
        return output.choices[0].message.content

    def get_analysis_steps(self, user_input: str) -> str:
        self.plotting_guide = self._get_plotting_guide(user_input)
        self.logger.info(f"\nPlotting guide recieved:\n{self.plotting_guide}")
        self.llm_output = self._get_plotting_steps(user_input)
        self.all_steps = check_output_format(self.llm_output, self.logger, "plot")
        self.logger.info(f"\nPlotting steps recieved:\n{self.all_steps}")
        self.preprocess_steps = None

        if "preprocess" in self.all_steps:
            self.preprocess_steps = self.all_steps["preprocess"]
        self.plotting_steps = self.all_steps["plot"]
        return self.preprocess_steps

    def _plot_subplot(
        self, plot_type: str, axes: np.ndarray, args: dict, df: pd.DataFrame, plot: dict
    ) -> None:
        if plot_type == "line":
            df.plot.line(
                x=plot.get("x"),
                y=plot.get("y"),
                ax=axes,
                **args,
            )
        elif plot_type == "bar":
            args["stacked"] = args.get("stacked", False)
            if isinstance(args["stacked"], str):
                if args["stacked"].lower() == "true":
                    args["stacked"] = True
                elif args["stacked"].lower() == "false":
                    args["stacked"] = False
                else:
                    self.logger.warning(
                        f"Invalid value provided for stacked: {args['stacked']}. Defaulting to False."
                    )
                    args["stacked"] = False
            if not isinstance(args["stacked"], bool):
                self.logger.warning(
                    f"Invalid value type provided for stacked: {type(args['stacked'])}. Defaulting to False."
                )
                args["stacked"] = False
            n_bars = 25
            if df[[plot.get("x"), plot.get("y")]].shape[0] > n_bars:
                self.logger.warning(
                    f"\nToo many bars given. Plotting only the first {n_bars} bars."
                )
                df_bar = df[[plot.get("x"), plot.get("y")]].head(n_bars)
            else:
                df_bar = df[[plot.get("x"), plot.get("y")]]

            df_bar.plot.bar(
                x=plot.get("x"),
                y=plot.get("y"),
                ax=axes,
                **args,
            )
        elif plot_type == "barh":
            df.plot.barh(
                x=plot.get("x"),
                y=plot.get("y"),
                ax=axes,
                **args,
            )
        elif plot_type == "scatter":
            df.plot.scatter(
                x=plot.get("x"),
                y=plot.get("y"),
                ax=axes,
                **args,
            )
        elif plot_type == "hist":
            df[plot.get("by")] = df[plot.get("by")].astype(float)
            df.plot.hist(
                by=plot.get("by"),
                ax=axes,
                **args,
            )
        else:
            raise ValueError(f"Invalid plot_type received: {plot_type}.")
        xlabel = plot.get("args", {}).get("xlabel")
        ylabel = plot.get("args", {}).get("ylabel")
        if xlabel:
            axes.set_xlabel(xlabel)
        if ylabel:
            axes.set_ylabel(ylabel)

    def _create_plot(self, plot_details: dict, df: pd.DataFrame) -> Plot:

        nrows, ncols = plot_details.get("subplots", (1, 1))
        fig, ax = plt.subplots(
            figsize=plot_details.get("figsize", (10, 10)),
            nrows=nrows,
            ncols=ncols,
        )
        if not isinstance(ax, np.ndarray):
            ax = np.array([ax])
        ax = ax.reshape((nrows, ncols))

        fig.suptitle(plot_details.get("title", "Plot"))
        for plot in plot_details.get("plots", []):
            self.logger.info(f"\nPlotting: {plot}")
            plot_type = plot.get("plot_type", "line")
            axes = ax[
                plot.get("subplot", (1, 1))[0] - 1, plot.get("subplot", (1, 1))[1] - 1
            ]
            args = plot.get("args", {})
            self._plot_subplot(plot_type, axes, args, df, plot)
        return fig

    def get_visualisation(self, df: pd.DataFrame) -> str:
        fig = self._create_plot(self.plotting_steps, df)
        dir_path = os.path.dirname(self.plot_path)
        if dir_path != "" and not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        fig.savefig(self.plot_path)
        plt.close(fig)
        self.logger.info(f"\nPlot saved at: {self.plot_path}\n")
        return self.plot_path
