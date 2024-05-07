# standard library imports
import os
import json
import logging
import traceback
from typing import Union

# third-party imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# local imports
from lyzr.base.llm import LiteLLM
from lyzr.data_analyzr.utils import (
    get_columns_names,
    format_df_details,
    iterate_llm_calls,
)
from lyzr.data_analyzr.plot_handler.utils import (
    handle_plotpath,
    convert_dtypes,
    flatten_list,
    set_analysis_attributes,
)
from lyzr.base.prompt import LyzrPromptFactory
from lyzr.base.base import UserMessage, SystemMessage
from lyzr.data_analyzr.models import FactoryBaseClass
from lyzr.data_analyzr.vector_store_utils import ChromaDBVectorStore
from testing.python_analysis_utils import PlottingStepsModel, PlotingStepsDetails
from lyzr.data_analyzr.analysis_handler import TxttoSQLFactory, PythonicAnalysisFactory


class PlotFactory(FactoryBaseClass):

    def __init__(
        self,
        llm: LiteLLM,
        logger: logging.Logger,
        context: str,
        plot_path: str,
        df_dict: dict,
        vector_store: ChromaDBVectorStore,
        analyzer: Union[PythonicAnalysisFactory, TxttoSQLFactory],
        analysis_output: Union[pd.DataFrame, list],
        max_retries: int = None,
        time_limit: int = None,
        auto_train: bool = None,
        **llm_kwargs,  # model_kwargs: dict
    ):
        super().__init__(
            llm=llm,
            logger=logger,
            context=context,
            vector_store=vector_store,
            max_retries=max_retries,
            time_limit=time_limit,
            auto_train=auto_train,
            llm_kwargs=llm_kwargs
        )
        self.analysis_output = analysis_output
        self._plot_path = handle_plotpath(plot_path, "png", self.logger)
        self._plotting_library = "matplotlib"
        self._plotting_guide = None
        self._rerun_analysis, self._analyzer, self._df_dict  = set_analysis_attributes(
            analysis_output = analysis_output,
            analyzer=analyzer,
            df_dict=df_dict,
        )

    def get_visualisation(self, user_input: str, **kwargs) -> str:
        if self._rerun_analysis:
            self.analysis_output = self._analyzer.run_analysis(
                user_input, for_plotting=True
            )
            self._rerun_analysis, self._analyzer, self._df_dict = set_analysis_attributes(
                analysis_output=self.analysis_output,
                analyzer=self._analyzer,
                df_dict=self._df_dict,
            )
        messages = self.get_prompt_messages(user_input)
        self.get_plotting_steps = iterate_llm_calls(
            max_retries=kwargs.pop(
                "max_retries", self.params.max_retries
            ),
            llm=self.llm,
            llm_messages=messages,
            logger=self.logger,
            log_messages={
                "start": f"Generating plotting steps for query: {user_input}",
                "end": f"Finished generation plotting steps for query: {user_input}",
            },
            time_limit=kwargs.pop("time_limit", self.params.time_limit),
        )(self.get_plotting_steps)
        self.steps = self.get_plotting_steps()
        self.fig = self.make_plot_figure(self.plot_df_dict[self.steps.df_name])
        if (
            kwargs.pop("auto_train", self.params.auto_train)
            and self.fig is not None
        ):
            self.add_training_data(user_input, json.dumps(self._steps.model_dump()))
        return self.save_plot_image()

    def get_prompt_messages(self, user_input: str) -> list:
        system_message_sections = ["context", "external_context", "task", "closing"]
        plotting_guide = ""
        self.plot_df_dict = {"dataset": self.analysis_output}
        schema = json.dumps(PlottingStepsModel.model_json_schema())
        if self._rerun_analysis:
            plotting_guide = self._make_plotting_guide(user_input)
            system_message_sections = [
                "context",
                "external_context",
                "guide",
                "task",
                "closing",
            ]
            self.plot_df_dict = self._df_dict
        messages = [
            LyzrPromptFactory(name="plotting_steps", prompt_type="system").get_message(
                use_sections=system_message_sections,
                plotting_lib=self.additional_kwargs["plotting_library"],
                schema=schema,
            ),
            LyzrPromptFactory(name="plotting_steps", prompt_type="user").get_message(
                question=user_input,
                df_details=format_df_details(self.plot_df_dict),
                guide=plotting_guide,
            ),
        ]
        question_examples_list = self.vector_store.get_similar_plotting_steps(user_input)
        for example in question_examples_list:
            if (example is not None) and ("question" in example) and ("steps" in example):
                messages.append(UserMessage(content=example["question"]))
                messages.append(SystemMessage(content=example["steps"]))
        return messages

    def _make_plotting_guide(self, user_input: str) -> str:
        plotting_guide_sections = ["context", "external_context", "task"]
        df_details = format_df_details(self.analysis_output)
        output = self.llm.run(
            messages=[
                LyzrPromptFactory(
                    name="plotting_guide", prompt_type="system"
                ).get_message(
                    use_sections=plotting_guide_sections,
                    context=self.context,
                    plotting_lib=self.additional_kwargs["plotting_library"],
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

    def get_plotting_steps(self, llm_response: str) -> PlottingStepsModel:
        steps = PlottingStepsModel(**json.loads(llm_response))
        return steps

    def make_plot_figure(self, df: pd.DataFrame) -> plt.Figure:
        nrows, ncols = self.steps.subplots
        fig, ax = plt.subplots(
            figsize=self.steps.figsize,
            nrows=nrows,
            ncols=ncols,
        )
        if not isinstance(ax, np.ndarray):
            ax = np.array([ax])
        ax = ax.reshape((nrows, ncols))

        fig.suptitle(self.steps.title)
        for plot in self.steps.plots:
            self._plot_subplot(
                plot_type=plot.plot_type,
                axes=ax[plot.subplot[0] - 1, plot.subplot[1] - 1],
                args=plot.args,
                df=df,
                plot=plot,
            )
        return fig

    def _plot_subplot(
        self,
        plot_type: str,
        axes: np.ndarray,
        args: dict,
        df: pd.DataFrame,
        plot: PlotingStepsDetails,
    ) -> None:
        columns_with_dtypes = {}
        if plot.x is not None:
            columns_with_dtypes[plot.x] = plot.x_dtype
        if plot.y is not None:
            columns_with_dtypes[plot.y] = plot.y_dtype
        if plot.by is not None:
            columns_with_dtypes[plot.by] = plot.by_dtype
        columns = get_columns_names(
            df_columns=df.columns,
            columns=list(flatten_list([plot.x, plot.y, plot.by])),
        )
        # convert column dtypes
        df = df.dropna(subset=columns)
        df = convert_dtypes(
            df=df,
            columns=columns_with_dtypes,
        ).infer_objects()

        if plot_type == "line":
            df.plot.line(
                x=plot.x,
                y=plot.y,
                ax=axes,
                **args,
            )
        elif plot_type == "bar":
            args["stacked"] = self._set_args_stacked(args)
            df_bar = self._get_bar_df(df, columns)
            df_bar.plot.bar(
                x=plot.x,
                y=plot.y,
                ax=axes,
                **args,
            )
        elif plot_type == "barh":
            args["stacked"] = self._set_args_stacked(args)
            df_bar = self._get_bar_df(df, columns)
            df_bar.plot.barh(
                x=plot.x,
                y=plot.y,
                ax=axes,
                **args,
            )
        elif plot_type == "scatter":
            df.plot.scatter(
                x=plot.x,
                y=plot.y,
                ax=axes,
                **args,
            )
        elif plot_type == "hist":
            df_hist = df[plot.by]
            df_hist.plot.hist(
                ax=axes,
                **args,
            )
        else:
            raise ValueError(f"Invalid plot_type received: {plot_type}.")
        xlabel = plot.args.get("xlabel")
        ylabel = plot.args.get("ylabel")
        if xlabel:
            axes.set_xlabel(xlabel)
        if ylabel:
            axes.set_ylabel(ylabel)

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
            stacked = False
        return stacked

    def _get_bar_df(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        n_bars = 25
        if df[columns].shape[0] > n_bars:
            num_cols = df[columns].select_dtypes(include=np.number).columns.tolist()
            df_bar = df[columns].sort_values(by=num_cols, ascending=False).head(n_bars)
        else:
            df_bar = df[columns]
        return df_bar

    def save_plot_image(self) -> str:
        plt.tight_layout()
        if not PlotFactory._savefig(self.fig, self.additional_kwargs["plot_path"]):
            self.logger.error(
                f"Error saving plot at: {self.additional_kwargs["plot_path"]}. Plot not saved. Displaying plot instead. Access the plot using `.fig` attribute.",
                extra={
                    "function": "save_plot_image",
                    "traceback": traceback.format_exc().splitlines(),
                }
            )
            plt.show()
        else:
            self.logger.info(f"\nPlot saved at: {self.additional_kwargs["plot_path"]}\n", extra={"function": "save_plot_image"})
            plt.close(self.fig)
        return self.additional_kwargs["plot_path"]

    @staticmethod
    def _savefig(fig: plt.Figure, path: str):
        try:
            dir_path = os.path.dirname(path)
            if dir_path.strip() != "":
                os.makedirs(dir_path, exist_ok=True)
            fig.savefig(path)
            return True
        except Exception:
            PlotFactory._savefig(fig, "generated_plots/plot.png")
        return False

    def add_training_data(self, user_input: str, steps: str):
        if steps is not None and steps.strip() != "":
            self.vector_store.add_training_plan(question=user_input, plot_steps=steps)
