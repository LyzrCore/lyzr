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
from lyzr.data_analyzr.utils import format_df_details, convert_to_numeric
from lyzr.data_analyzr.output_handler import check_output_format


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

        self.plot_path = plot_path
        if not os.path.isfile(self.plot_path):
            dir_path = os.path.dirname(self.plot_path)
            if dir_path.strip() != "":
                os.makedirs(dir_path, exist_ok=True)
            if os.path.isdir(self.plot_path):
                self.plot_path = os.path.join(self.plot_path, "plot.png")
            else:
                self.logger.warn(
                    f'Incorrect path for plot image provided: {self.plot_path}. Defaulting to "generated_plots/plot.png".'
                )
                self.plot_path = "generated_plots/plot.png"
        if os.path.splitext(self.plot_path)[1] != ".png":
            self.plot_path = os.path.join(os.path.splitext(self.plot_path)[0], ".png")

    def _get_plotting_guide(self, user_input: str) -> str:
        self.model.set_messages(
            messages=[
                {
                    "role": "system",
                    "content": Prompt("analysis_guide_pt")
                    .format(
                        df_details=format_df_details(self.df_dict, self.df_info_dict),
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
                        df_details=format_df_details(self.df_dict, self.df_info_dict),
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
        columns = [plot.get("x"), plot.get("y")]
        columns.extend(plot.get("by", []))
        df = convert_to_numeric(df, columns=columns).infer_objects()

        if plot_type == "line":
            self.logger.info(f"\nDF to be plot:\n{df.head()}\n")
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
            columns = [plot.get("x"), plot.get("y")]
            if df[columns].shape[0] > n_bars:
                self.logger.warning(
                    f"\nToo many bars given. Plotting only the top {n_bars} bars."
                )
                num_cols = df[columns].select_dtypes(include=np.number).columns.tolist()
                df_bar = (
                    df[columns].sort_values(by=num_cols, ascending=False).head(n_bars)
                )
            else:
                df_bar = df[columns]

            self.logger.info(f"\nDF to be plot:\n{df_bar.head()}\n")
            df_bar.plot.bar(
                x=plot.get("x"),
                y=plot.get("y"),
                ax=axes,
                **args,
            )
        elif plot_type == "barh":
            self.logger.info(f"\nDF to be plot:\n{df.head()}\n")
            df.plot.barh(
                x=plot.get("x"),
                y=plot.get("y"),
                ax=axes,
                **args,
            )
        elif plot_type == "scatter":
            self.logger.info(f"\nDF to be plot:\n{df.head()}\n")
            df.plot.scatter(
                x=plot.get("x"),
                y=plot.get("y"),
                ax=axes,
                **args,
            )
        elif plot_type == "hist":
            df_hist = df[plot.get("by")]
            self.logger.info(f"\nDF to be plot:\n{df_hist.head()}\n")
            df_hist.plot.hist(
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

    def _create_plot(self, plot_details: dict, df: pd.DataFrame) -> plt.Figure:

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
        plt.tight_layout()
        fig.savefig(self.plot_path)
        plt.close(fig)
        self.logger.info(f"\nPlot saved at: {self.plot_path}\n")
        return self.plot_path
