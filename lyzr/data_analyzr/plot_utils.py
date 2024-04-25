# standard library imports
import os
import time
import logging
import traceback
from typing import Union

# third-party imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# local imports
from lyzr.base.base import SystemMessage, AssistantMessage
from lyzr.base.prompt import LyzrPromptFactory
from lyzr.base.llm import LiteLLM
from lyzr.data_analyzr.data_connector import DataConnector
from lyzr.data_analyzr.txt_to_sql_utils import TxttoSQLFactory
from lyzr.data_analyzr.output_handler import check_output_format
from lyzr.data_analyzr.ml_analysis_utils import MLAnalysisFactory
from lyzr.data_analyzr.utils import (
    convert_to_numeric,
    flatten_list,
    get_columns_names,
    format_df_with_describe,
)


class PlotFactory:

    def __init__(
        self,
        model: LiteLLM,
        logger: logging.Logger,
        plot_context: str,
        plot_path: str,
        df_dict: dict,
        database_connector: DataConnector,
        analysis_type: str,
        analyzer: Union[MLAnalysisFactory, TxttoSQLFactory],
        analysis_output: Union[pd.DataFrame, list],
    ):
        self.model = model
        self.model.set_model_kwargs(
            model_kwargs=dict(seed=123, temperature=0.1, top_p=0.5)
        )
        if not isinstance(analysis_output, pd.DataFrame) or analysis_output.size < 2:
            self.use_analysis = True
            self.analysis_type = analysis_type
            self.analyzer = analyzer
            self.df_dict = df_dict
            self.database_connector = database_connector
            self.analysis_output = analysis_output
        else:
            self.use_analysis = False
            self.analysis_output = analysis_output
            self.df_dict = None
            self.database_connector = None
            self.analyzer = None
            self.analysis_type = None
        self.logger = logger
        self.context = (
            plot_context.strip() + "\n\n" if plot_context.strip() != "" else ""
        )
        self.plotting_library = "matplotlib"
        self.output_format = "png"
        self.plot_path = self._handle_plotpath(plot_path)

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
        plotting_guide_sections = ["context", "external_context"]
        if self.use_analysis and (self.analysis_type == "ml"):
            plotting_guide_sections.append("task_with_analysis")
            df_details = format_df_with_describe(self.df_dict)
        else:
            plotting_guide_sections.append("task_no_analysis")
            df_details = format_df_with_describe(self.analysis_output)
        output = self.model.run(
            messages=[
                LyzrPromptFactory(
                    name="plotting_guide", prompt_type="system"
                ).get_message(
                    use_sections=plotting_guide_sections,
                    context=self.context,
                    plotting_lib=self.plotting_library,
                ),
                LyzrPromptFactory(
                    name="plotting_guide", prompt_type="user"
                ).get_message(
                    df_details=df_details,
                    question=user_input,
                ),
            ],
            max_tokens=500,
        )
        return output.message.content.strip()

    def _get_plotting_steps_no_analysis_messages(self, user_input: str) -> list:
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
        messages = [
            LyzrPromptFactory(name="plotting_steps", prompt_type="system").get_message(
                use_sections=["context", "task_no_analysis", "closing"],
                plotting_lib=self.plotting_library,
                schema=schema,
            ),
            LyzrPromptFactory(name="plotting_steps", prompt_type="user").get_message(
                question=user_input,
                df_details=format_df_with_describe(self.analysis_output),
                guide=self.plotting_guide,
            ),
        ]
        return messages

    def _get_plotting_steps_with_analysis_messages(self, user_input: str) -> list:
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
        messages = [
            LyzrPromptFactory(name="plotting_steps", prompt_type="system").get_message(
                use_sections=["context", "task_with_analysis", "closing"],
                plotting_lib=self.plotting_library,
                schema=schema,
            ),
            LyzrPromptFactory(name="plotting_steps", prompt_type="user").get_message(
                question=user_input,
                df_details=format_df_with_describe(self.analysis_output),
                guide=self.plotting_guide,
            ),
        ]
        return messages

    def retry_plotting_steps(
        self,
        messages: list,
    ) -> bool:
        for _ in range(5):
            try:
                llm_output = self.model.run(
                    messages=messages,
                    response_format={"type": "json_object"},
                    max_tokens=2000,
                    top_p=1,
                ).message.content
                messages.append(AssistantMessage(content=llm_output))
                self.all_steps = check_output_format(llm_output, self.logger, "plot")
                self.logger.info(f"\nPlotting steps recieved:\n{self.all_steps}")
                self.preprocess_steps = self.all_steps.get("preprocess", None)
                if self.preprocess_steps is not None and len(self.preprocess_steps) > 0:
                    _, self.analysis_output = self.analyzer.get_analysis_from_steps(
                        self.preprocess_steps
                    )
                    self.plotting_steps = self.all_steps["plot"]
                else:
                    self.plotting_steps = self.all_steps
                self.fig = self._create_plot(self.analysis_output)
                return True
            except RecursionError:
                plt.close("all")
                raise RecursionError(
                    "The request could not be completed. Please wait a while and try again."
                )
            except Exception as e:
                plt.close("all")
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
        return False

    def _set_args_stacked(self, args: dict) -> bool:
        stacked = args.get("stacked", False)
        if isinstance(stacked, str):
            if stacked.lower() == "true":
                stacked = True
            elif stacked.lower() == "false":
                stacked = False
            else:
                stacked = False
        if not isinstance(stacked, bool):
            self.logger.warning(
                f"Invalid value type provided for stacked: {type(args['stacked'])}. Defaulting to False."
            )
            stacked = False
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

    def _create_plot(self, df: pd.DataFrame) -> plt.Figure:
        nrows, ncols = self.plotting_steps.get("subplots", (1, 1))
        fig, ax = plt.subplots(
            figsize=self.plotting_steps.get("figsize", (10, 10)),
            nrows=nrows,
            ncols=ncols,
        )
        if not isinstance(ax, np.ndarray):
            ax = np.array([ax])
        ax = ax.reshape((nrows, ncols))

        fig.suptitle(self.plotting_steps.get("title", "Plot"))
        for plot in self.plotting_steps.get("plots", []):
            self.logger.info(f"\nPlotting: {plot}")
            plot_type = plot.get("plot_type", "line")
            axes = ax[
                plot.get("subplot", (1, 1))[0] - 1, plot.get("subplot", (1, 1))[1] - 1
            ]
            args = plot.get("args", {})
            self._plot_subplot(plot_type, axes, args, df, plot)
        return fig

    def get_visualisation_image(self) -> str:
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

    def get_visualisation(self, user_input: str) -> str:
        self.start_time = time.time()
        self.plotting_guide = self._get_plotting_guide(user_input)
        self.logger.info(f"\nPlotting guide recieved:\n{self.plotting_guide}")
        if self.use_analysis:
            if self.analysis_type == "ml":
                messages = self._get_plotting_steps_with_analysis_messages(user_input)
            elif self.analysis_type == "sql":
                self.analysis_output = self.analyzer.run_analysis_for_plotting(
                    user_input=user_input
                )
                messages = self._get_plotting_steps_no_analysis_messages(user_input)
        else:
            messages = self._get_plotting_steps_no_analysis_messages(user_input)
        if self.retry_plotting_steps(messages):
            self.get_visualisation_image()
            return self.plot_path
