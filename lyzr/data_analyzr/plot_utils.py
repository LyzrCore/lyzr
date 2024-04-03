# standard library imports
import io
import os
import logging

# third-party imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# local imports
from lyzr.base.prompt import Prompt
from lyzr.base.llms import LLM, set_model_params
from lyzr.data_analyzr.output_handler import check_output_format
from lyzr.data_analyzr.utils import (
    format_df_details,
    convert_to_numeric,
    flatten_list,
    get_columns_names,
)


class PlotFactory:
    def __init__(
        self,
        plotting_model: LLM,
        plotting_model_kwargs: dict,
        df_dict: list[pd.DataFrame],
        logger: logging.Logger,
        plot_context: str,
        plot_path: str,
        use_guide: bool = True,
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

        self.plot_path = self._handle_plotpath(plot_path)
        self.use_guide = use_guide

    def _handle_plotpath(self, plot_path) -> str:
        plot_path = PlotFactory._fix_plotpath(plot_path)
        try:
            open(plot_path, "w").close()
            return plot_path
        except Exception:
            self.logger.warning(
                f'Incorrect path for plot image provided: {self.plot_path}. Defaulting to "generated_plots/plot.png".'
            )
            return self._handle_plotpath("generated_plots/plot.png")

    @staticmethod
    def _fix_plotpath(plot_path: str) -> str:
        if os.path.isdir(plot_path):
            plot_path = os.path.join(plot_path, "plot.png")
        if os.path.splitext(plot_path)[1] != ".png":
            plot_path = os.path.splitext(plot_path)[0] + ".png"
        dir_path = os.path.dirname(plot_path)
        if dir_path.strip() != "":
            os.makedirs(dir_path, exist_ok=True)
        return plot_path

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
        }
        self.model.set_messages(
            messages=[
                {
                    "role": "system",
                    "content": Prompt("plotting_steps_pt")
                    .format(
                        plotting_lib=self.plotting_library,
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

    def _get_plotting_steps_from_guide(self, user_input: str) -> str:
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
                    "content": Prompt("plotting_steps_with_analysis_pt")
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
        if self.use_guide:
            self.plotting_guide = self._get_plotting_guide(user_input)
            self.logger.info(f"\nPlotting guide recieved:\n{self.plotting_guide}")
            self.llm_output = self._get_plotting_steps_from_guide(user_input)
            self.all_steps = check_output_format(self.llm_output, self.logger, "plot")
            self.logger.info(f"\nPlotting steps recieved:\n{self.all_steps}")
            self.preprocess_steps = None

            if "preprocess" in self.all_steps:
                self.preprocess_steps = self.all_steps["preprocess"]
            self.plotting_steps = self.all_steps["plot"]
            return self.preprocess_steps
        else:
            self.all_steps = None
            self.preprocess_steps = None
            self.llm_output = self._get_plotting_steps(user_input)
            self.plotting_steps = check_output_format(
                self.llm_output, self.logger, "plot"
            )
            self.logger.info(f"\nPlotting steps recieved:\n{self.plotting_steps}")
            return self.preprocess_steps

    def _set_args_stacked(self, args: dict) -> bool:
        stacked = args.get("stacked", False)
        if isinstance(stacked, str):
            if stacked.lower() == "true":
                stacked = True
            elif stacked.lower() == "false":
                stacked = False
            else:
                stacked = False
        if not isinstance(args["stacked"], bool):
            self.logger.warning(
                f"Invalid value type provided for stacked: {type(args['stacked'])}. Defaulting to False."
            )
            args["stacked"] = False
        return stacked

    def _get_bar_df(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        n_bars = 25
        if df[columns].shape[0] > n_bars:
            self.logger.warning(
                f"\nToo many bars given. Plotting only the top {n_bars} bars."
            )
            num_cols = df[columns].select_dtypes(include=np.number).columns.tolist()
            df_bar = df[columns].sort_values(by=num_cols, ascending=False).head(n_bars)
        else:
            df_bar = df[columns]
        return df_bar

    def _plot_subplot(
        self, plot_type: str, axes: np.ndarray, args: dict, df: pd.DataFrame, plot: dict
    ) -> None:
        if "x" in plot and isinstance(plot["x"], list):
            plot["x"] = plot["x"][0]
        if "y" in plot and isinstance(plot["y"], list):
            plot["y"] = plot["y"][0]
        columns = get_columns_names(
            df_columns=df.columns,
            columns=list(
                flatten_list([plot.get("x", []), plot.get("y", []), plot.get("by", [])])
            ),
            logger=self.logger,
        )
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
            args["stacked"] = self._set_args_stacked(args)
            df_bar = self._get_bar_df(df, columns)
            self.logger.info(f"\nDF to be plot:\n{df_bar.head()}\n")
            df_bar.plot.bar(
                x=plot.get("x"),
                y=plot.get("y"),
                ax=axes,
                **args,
            )
        elif plot_type == "barh":
            args["stacked"] = self._set_args_stacked(args)
            df_bar = self._get_bar_df(df, columns)
            self.logger.info(f"\nDF to be plot:\n{df_bar.head()}\n")
            df_bar.plot.barh(
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
        self.fig = self._create_plot(self.plotting_steps, df)
        plt.tight_layout()
        if not PlotFactory._savefig(self.fig, self.plot_path, self.logger):
            self.logger.error(
                f"Error saving plot at: {self.plot_path}. Plot not saved. Displaying plot instead. Access the plot using `.fig` attribute."
            )
            plt.show()
        else:
            self.logger.info(f"\nPlot saved at: {self.plot_path}\n")
            plt.close(self.fig)
        return self.plot_path

    @staticmethod
    def _savefig(fig, path, logger):
        try:
            dir_path = os.path.dirname(path)
            if dir_path.strip() != "":
                os.makedirs(dir_path, exist_ok=True)
            fig.savefig(path)
            return True
        except Exception:
            logger.error(
                f"Error saving plot at: {path}. Trying to save at default location: 'generated_plots/plot.png'."
            )
            PlotFactory._savefig(fig, "generated_plots/plot.png", logger)
        return False
