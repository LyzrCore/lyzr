# standard library imports
import os
import time
import traceback
from typing import Union, Literal, Optional, Any

# third-party imports
import numpy as np
import pandas as pd

# local imports
from lyzr.base.prompt import Prompt
from lyzr.data_analyzr.file_utils import get_db_details
from lyzr.data_analyzr.txt_to_sql_utils import TxttoSQLFactory
from lyzr.data_analyzr.ml_analysis_utils import MLAnalysisFactory
from lyzr.base.errors import (
    MissingValueError,
)
from lyzr.base.llms import (
    LLM,
    get_model,
    set_model_params,
)
from lyzr.data_analyzr.plot_utils import PlotFactory
from lyzr.data_analyzr.utils import (
    format_df_with_describe,
    format_df_with_info,
    get_info_dict_from_df_dict,
)

# imports for logging
import sys
import logging

# imports for legacy usage
from PIL import Image
from pathlib import Path
from pandas.errors import EmptyDataError


class DataAnalyzr:

    def __init__(
        self,
        analysis_type: Optional[Literal["sql", "ml", "skip"]] = None,
        api_key: Optional[str] = None,
        gen_model: Optional[Union[dict, LLM]] = None,
        analysis_model: Optional[Union[dict, LLM]] = None,
        plot_model: Optional[Union[dict, LLM]] = None,
        user_input: Optional[str] = None,
        context: Optional[str] = None,
        log_level: Optional[
            Literal["NOTSET", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        ] = "INFO",
        print_log: Optional[bool] = False,
        log_filename: Optional[str] = "dataanalyzr.log",
        # legacy usage
        df: Union[str, pd.DataFrame] = None,
        model: Optional[LLM] = None,
        model_type: Optional[str] = None,
        model_name: Optional[str] = None,
        seed: int = None,
    ):
        self.api_key = (
            api_key or os.environ.get("API_KEY") or os.environ.get("OPENAI_API_KEY")
        )
        if self.api_key is None:
            raise MissingValueError("api key")

        self.user_input = user_input
        self.log_filename = log_filename
        self._set_logger(log_level, print_log)

        if df is not None:
            # legacy usage
            self._legacy_usage(df, user_input, model, model_type, model_name, seed)
        else:
            if analysis_type is None:
                raise MissingValueError("analysis_type")
            if gen_model is None:
                gen_model = {
                    "type": "openai",
                    "name": "gpt-4-1106-preview",
                    "kwargs": {},
                }
            if isinstance(gen_model, LLM):
                self._gen_model = gen_model
                self._gen_model_kwargs = {}
            else:
                self._gen_model_kwargs = gen_model["kwargs"]
                self._gen_model = get_model(
                    self.api_key,
                    gen_model["type"],
                    gen_model["name"],
                    **self._gen_model_kwargs,
                )

            self.analysis_type = analysis_type.lower().strip()
            self._analysis_model = analysis_model
            self._plot_model = plot_model
            self.context = context or ""

        (
            self.dataset_description_output,
            self.ai_queries,
            self.analysis_output,
            self.visualisation_output,
            self.insights_output,
            self.recommendations_output,
            self.tasks_output,
        ) = (None, None, None, None, None, None, None)

    def _legacy_usage(
        self,
        df: Union[str, pd.DataFrame],
        user_input: Optional[str],
        model: Optional[LLM],
        model_type: Optional[str],
        model_name: Optional[str],
        seed: int,
    ):
        self.logger.warning(
            "The `df` parameter is deprecated and will be removed in a future version. Please use the `get_data` method to load data."
        )
        for param in ["model", "model_type", "model_name", "seed"]:
            if locals()[param] is not None:
                self.logger.warning(
                    f"The `{param}` parameter is deprecated and will be removed in a future version. Please use the `analysis_model` parameter to set the analysis model, and the `gen_model` parameter to set the generation model."
                )
        self._gen_model = model or get_model(
            api_key=self.api_key,
            model_type=model_type or os.environ.get("MODEL_TYPE", "openai"),
            model_name=model_name or os.environ.get("MODEL_NAME", "gpt-4-1106-preview"),
        )
        self._analysis_model = self._plot_model = (
            model
            or os.environ.get("LLM_MODEL")
            or get_model(
                api_key=self.api_key,
                model_type=model_type or os.environ.get("MODEL_TYPE", "openai"),
                model_name=model_name or os.environ.get("MODEL_NAME", "gpt-3.5-turbo"),
            )
        )
        self._gen_model_kwargs = self._analysis_model_kwargs = (
            self._plot_model_kwargs
        ) = {"seed": seed}
        self.context = ""
        self.analysis_type = "ml"
        self.user_input = user_input

        def _clean_df(df: pd.DataFrame):
            df = df[df.columns[df.isnull().mean() < 0.5]]
            cat_columns = df.select_dtypes(include=["object"]).columns
            num_columns = df.select_dtypes(include=[np.number]).columns
            df[cat_columns] = df[cat_columns].apply(lambda x: x.fillna(x.mode()[0]))
            df[num_columns] = df[num_columns].apply(lambda x: x.fillna(x.mean()))
            df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
            df = df.drop_duplicates(keep="first")
            return df

        if isinstance(df, str):
            self.database_connector, self.df_dict, self.vector_store = get_db_details(
                "ml", "files", {"datasets": {"Dataset": df}}, {}, logger=self.logger
            )
            for name, df in self.df_dict.items():
                if not isinstance(df, pd.DataFrame):
                    continue
                self.df_dict[name] = _clean_df(df)
        elif isinstance(df, pd.DataFrame):
            if df.empty:
                raise EmptyDataError("The provided DataFrame is empty.")
            self.df_dict = {"dataset": _clean_df(df)}
            self.database_connector = None
            self.vector_store = None
        else:
            raise ValueError("df must be a path to a file or a pd.DataFrame object.")
        self.df_info_dict = get_info_dict_from_df_dict(self.df_dict)

    def _set_logger(self, log_level, print_log):
        self.logger = logging.getLogger(__name__)
        numeric_level = getattr(logging, log_level.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError("Invalid log level: %s" % log_level)
        self.logger.setLevel(numeric_level)

        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        if print_log:
            # output logs to stdout
            handler = logging.StreamHandler(sys.stdout)
            handler.setLevel(numeric_level)
            self.logger.addHandler(handler)

        log_filename = self.log_filename
        file_handler = logging.FileHandler(
            log_filename, mode="a"
        )  # Open the log file in append mode
        file_handler.setLevel(numeric_level)

        # Optionally, you can set a formatter for the file handler if you want a different format for file logs
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s\n%(message)s\n",
            datefmt="%d-%b-%y %H:%M:%S",
        )
        formatter.converter = time.gmtime
        file_handler.setFormatter(formatter)

        # Add the file handler to the logger
        self.logger.addHandler(file_handler)

    # Function to get datasets
    def get_data(
        self,
        db_type: Literal[
            "files",
            "redshift",
            "postgres",
            "sqlite",
        ],
        config: dict,
        # if db_type == "files", config_keys = datasets, files_kwargs, db_path
        # if db_type == "redshift" or "postgres" or "snowflake" or "mysql", config_keys = host, port, user, password, database, schema, table
        # if db_type == "snowflake", config_keys = warehouse, role
        # if db_type == "sqlite", config_keys = db_path
        vector_store_config,
    ) -> Any:
        self.database_connector, self.df_dict, self.vector_store = get_db_details(
            self.analysis_type, db_type, config, vector_store_config, self.logger
        )
        self.df_info_dict = (
            get_info_dict_from_df_dict(self.df_dict)
            if self.df_dict is not None
            else None
        )

    def analysis(
        self,
        user_input: str,
        analysis_context: str,
        analysis_steps: dict = None,
    ):
        if self.analysis_type == "skip" and analysis_steps is None:
            if self.df_dict is None:
                self.logger.info(
                    "No analysis performed. Fetching dataframes from database."
                )
                self.df_dict = self.database_connector.fetch_dataframes_dict()
            self.analysis_output = self.df_dict
            self.analysis_guide = (
                "No analysis performed. Analysis output is the given dataframe."
            )
            return self.analysis_output
        self._analysis_model = self._analysis_model or {
            "type": "openai",
            "name": "gpt-3.5-turbo-1106",
            "kwargs": {},
        }
        if isinstance(self._analysis_model, LLM):
            self._analysis_model_kwargs = {}
        elif isinstance(self._analysis_model, dict):
            self._analysis_model_kwargs = self._analysis_model.get("kwargs", {})
            self._analysis_model = get_model(
                self.api_key,
                self._analysis_model.get("type"),
                self._analysis_model.get("name"),
                **self._analysis_model_kwargs,
            )
        if self.analysis_type == "sql" and analysis_steps is None:
            return self._txt_to_sql_analysis(user_input, analysis_context)
        if self.analysis_type == "ml" or analysis_steps is not None:
            return self._ml_analysis(
                user_input, analysis_context, analysis_steps=analysis_steps
            )

    def _ml_analysis(
        self, user_input: str, analysis_context: str = None, analysis_steps: dict = None
    ):
        self.analyzer = MLAnalysisFactory(
            model=self._analysis_model,
            data_dict=self.df_dict,
            data_info_dict=self.df_info_dict,
            logger=self.logger,
            context=analysis_context,
            model_kwargs=self._analysis_model_kwargs,
        )
        if analysis_steps is not None:
            _, data = self.analyzer.run_analysis(analysis_steps)
            return data
        else:
            self.analysis_output = self.analyzer.get_analysis_output(user_input)
            self.analysis_guide = self.analyzer.analysis_guide
            return self.analysis_output

    def _txt_to_sql_analysis(self, user_input: str, analysis_context: str = None):
        self.analyzer = TxttoSQLFactory(
            model=self._analysis_model,
            db_connector=self.database_connector,
            logger=self.logger,
            context=analysis_context,
            model_kwargs=self._analysis_model_kwargs,
            vector_store=self.vector_store,
        )
        self.analysis_output = self.analyzer.get_analysis_output(user_input)
        self.analysis_guide = self.analyzer.analysis_guide
        return self.analysis_output

    def visualisation(
        self,
        user_input: str,
        plot_context: str = None,
        plot_path: str = None,
    ):
        if self.user_input == user_input and self.visualisation_output is not None:
            return self.visualisation_output

        if plot_path is None:
            plot_path = Path("generated_plots/plot.png").as_posix()
        else:
            plot_path = Path(plot_path).as_posix()

        if self.df_dict is None:
            self.logger.info("Fetching dataframes from database to make visualization.")
            self.df_dict = self.database_connector.fetch_dataframes_dict()

        plot_context = plot_context or self.context
        self.user_input = user_input or self.user_input
        if self.user_input is None:
            raise MissingValueError(["user_input"])

        self._plot_model = self._plot_model or {
            "type": "openai",
            "name": "gpt-3.5-turbo-1106",
            "kwargs": {},
        }
        if isinstance(self._plot_model, LLM):
            self._plot_model_kwargs = {}
        elif isinstance(self._plot_model, dict):
            self._plot_model_kwargs = self._plot_model.get("kwargs", {})
            self._plot_model = get_model(
                self.api_key,
                self._plot_model.get("type"),
                self._plot_model.get("name"),
                **self._plot_model_kwargs,
            )

        self.visualisation_output = None
        self.start_time = time.time()
        while True:
            try:
                self.logger.info("Generating visualisation\n")
                plotter = PlotFactory(
                    plotting_model=self._plot_model,
                    plotting_model_kwargs=self._plot_model_kwargs,
                    df_dict=self.df_dict,
                    logger=self.logger,
                    plot_context=plot_context,
                    plot_path=plot_path,
                )
                analysis_steps = plotter.get_analysis_steps(self.user_input)
                if analysis_steps is not None and "steps" in analysis_steps:
                    if len(analysis_steps["steps"]) == 0:
                        self.plot_df = self.df_dict[analysis_steps["df_name"]]
                    else:
                        self.plot_df = self.analysis(user_input, "", analysis_steps)
                else:
                    self.logger.info(
                        "No analysis steps found. Using first dataframe for plotting.\n"
                    )
                    self.plot_df = self.df_dict[list(self.df_dict.keys())[0]]

                self.visualisation_output = plotter.get_visualisation(self.plot_df)
                return self.visualisation_output
            except RecursionError:
                raise RecursionError(
                    "The request could not be completed. Please wait a while and try again."
                )
            except Exception as e:
                if time.time() - self.start_time > 30:
                    raise TimeoutError(
                        "The request could not be completed. Please wait a while and try again."
                    )
                self.logger.info(f"{e.__class__.__name__}: {e}")
                self.logger.info("Traceback:\n{}\n".format(traceback.format_exc()))
                continue

    def insights(
        self,
        user_input: str,
        insights_context: Optional[str] = None,
        n_insights: Optional[int] = 3,
    ) -> str:
        if "analysis_guide" not in self.__dict__:
            self.analysis_guide = ""
        self.logger.info("Generating insights\n")
        self._gen_model.set_messages(
            messages=[
                {
                    "role": "system",
                    "content": Prompt("insights_context_pt")
                    .format(context=insights_context, n_insights=n_insights)
                    .text,
                },
                {
                    "content": Prompt("insights_pt")
                    .format(
                        user_input=user_input,
                        analysis_context=self.analysis_guide,
                        analysis_output=format_df_with_describe(self.analysis_output),
                        date=time.strftime("%d %b %Y"),
                    )
                    .text,
                    "role": "user",
                },
            ]
        )
        self._gen_model_kwargs = set_model_params(
            {
                "temperature": 0.3,
                "top_p": 1,
                "frequency_penalty": 0,
                "presence_penalty": 0,
            },
            self._gen_model_kwargs,
        )
        self.insights_output = (
            self._gen_model.run(**self._gen_model_kwargs).choices[0].message.content
        )
        return self.insights_output

    def recommendations(
        self,
        user_input: Optional[str] = None,
        use_insights: Optional[bool] = True,
        recs_format: Optional[dict] = None,
        recommendations_context: Optional[str] = None,
        n_recommendations: Optional[int] = 3,
        output_type: Optional[Literal["text", "json"]] = "text",
        # legacy usage
        insights: Optional[str] = None,
        schema: Optional[list] = None,
    ) -> str:
        if insights is not None or schema is not None:
            use_insights = True
            output_type = "json"
        if insights is not None:
            self.logger.warning(
                "The `insights` parameter is deprecated and will be removed in a future version. Please use the `insights_output` attribute to set the insights."
            )
            self.insights_output = insights
        if schema is not None:
            self.logger.warning(
                "The `schema` parameter is deprecated and will be removed in a future version. Please use the `recs_format` parameter to set the recommendations format as a dictionary."
            )
            for elem in schema:
                if isinstance(elem, dict):
                    recs_format = elem
                    break

        if recommendations_context is None or recommendations_context == "":
            recommendations_context = Prompt("rectxt_default_context_pt").text

        if not use_insights:
            insights = None
            recommendations_context += Prompt("rectxt_task_no_insights_pt").text
        else:
            insights = self.insights_output
            recommendations_context += Prompt("rectxt_task_with_insights_pt").text

        if output_type.lower().strip() == "json":
            recs_format = recs_format or {
                "Recommendation": "string",
                "Basis of the Recommendation": "string",
                "Impact if implemented": "string",
            }
            recommendations_context += (
                Prompt("rectxt_json_output_pt")
                .format(
                    output_json_keys=", ".join(recs_format.keys()),
                    json_schema=[recs_format for _ in range(n_recommendations)],
                )
                .text
            )
            self._gen_model_kwargs["response_format"] = {"type": "json_object"}
        elif output_type.lower().strip() == "text":
            recommendations_context += (
                Prompt("rectxt_text_output_pt")
                .format(count_recommendations=n_recommendations)
                .text
            )
        recommendations_context += Prompt("rectxt_closing_text_pt").text

        recommendations_pt = (
            Prompt("recommendations_pt")
            .format(
                user_input=f"User query: {user_input}",
                insights=(
                    "" if insights is None else f"\nAnalysis insights:\n{insights}\n"
                ),
                analysis_output=f"Analysis output:\n{format_df_with_describe(self.analysis_output)}",
            )
            .text
        )
        self.logger.info("Generating recommendations\n")
        self._gen_model.set_messages(
            messages=[
                {
                    "content": recommendations_context,
                    "role": "system",
                },
                {
                    "content": recommendations_pt,
                    "role": "user",
                },
            ]
        )
        self._gen_model_kwargs = set_model_params(
            {
                "temperature": 0.3,
                "top_p": 1,
                "frequency_penalty": 0,
                "presence_penalty": 0,
            },
            self._gen_model_kwargs,
        )
        self.recommendations_output = (
            self._gen_model.run(**self._gen_model_kwargs).choices[0].message.content
        )
        return self.recommendations_output

    def tasks(
        self,
        user_input: Optional[str] = None,
        tasks_context: Optional[str] = None,
        n_tasks: Optional[int] = 3,
        # legacy usage
        insights: Optional[str] = None,
        recommendations: Optional[str] = None,
    ) -> str:
        if insights is not None:
            self.logger.warning(
                "The `insights` parameter is deprecated and will be removed in a future version. Please use the `insights_output` attribute to set the insights."
            )
            self.insights_output = insights
        if recommendations is not None:
            self.logger.warning(
                "The `recommendations` parameter is deprecated and will be removed in a future version. Please use the `recommendations_output` attribute to set the recommendations."
            )
            self.recommendations_output = recommendations
        self.logger.info("Generating tasks\n")
        self._gen_model.set_messages(
            messages=[
                {
                    "content": Prompt("tasks_context_pt")
                    .format(context=tasks_context, n_tasks=n_tasks)
                    .text,
                    "role": "system",
                },
                {
                    "content": Prompt("tasks_pt")
                    .format(
                        user_input=user_input or self.user_input,
                        insights=self.insights_output,
                        recommendations=self.recommendations_output,
                    )
                    .text,
                    "role": "user",
                },
            ]
        )
        self._gen_model_kwargs = set_model_params(
            {
                "temperature": 1,
                "top_p": 0.3,
                "frequency_penalty": 0.7,
                "presence_penalty": 0.3,
            },
            self._gen_model_kwargs,
        )
        self.tasks_output = (
            self._gen_model.run(**self._gen_model_kwargs).choices[0].message.content
        )
        return self.tasks_output

    def ask(
        self,
        user_input: str = None,
        outputs: list[
            Literal["visualisation", "insights", "recommendations", "tasks"]
        ] = None,
        plot_path: str = None,
        rerun_analysis: bool = True,
        use_insights: bool = True,
        recs_format: dict = None,
        recs_output_type: Literal["text", "json"] = None,
        counts: dict = None,  # keys: insights, recommendations, tasks
        context: dict = None,  # keys: analysis, insights, recommendations, tasks
    ):
        if use_insights is None:
            use_insights = True
        if rerun_analysis is None:
            rerun_analysis = True
        if outputs is None:
            outputs = ["visualisation", "insights", "recommendations", "tasks"]
        if user_input is None and self.user_input is None:
            raise MissingValueError(["user_input"])
        if user_input is None:
            user_input = self.user_input
        context = context or {}
        counts = counts or {}

        if self.analysis_output is None or rerun_analysis:
            self.analysis_output = self.analysis(
                user_input=user_input,
                analysis_context=context.get("analysis", self.context),
            )

        if "visualisation" in outputs:
            self.visualisation_output = self.visualisation(
                user_input=user_input,
                plot_context=context.get("visualisation", self.context),
                plot_path=plot_path,
            )
        else:
            self.visualisation_output = ""

        if (
            "insights" in outputs
            or "tasks" in outputs
            or ("recommendations" in outputs and use_insights)
        ):
            self.insights_output = self.insights(
                user_input=user_input,
                insights_context=context.get("insights", self.context),
                n_insights=counts.get("insights", 3),
            )
        else:
            self.insights_output = ""

        if "recommendations" in outputs or "tasks" in outputs:
            self.recommendations_output = self.recommendations(
                user_input=user_input,
                use_insights=use_insights,
                recs_format=recs_format,
                recommendations_context=context.get("recommendations", self.context),
                n_recommendations=counts.get("recommendations", 3),
                output_type=recs_output_type or "text",
            )
        else:
            self.recommendations_output = ""

        if "tasks" in outputs:
            self.tasks_output = self.tasks(
                user_input=user_input,
                tasks_context=context.get("tasks", self.context),
                n_tasks=counts.get("tasks", 5),
            )
        else:
            self.tasks_output = ""

        return {
            "visualisation": self.visualisation_output,
            "insights": self.insights_output,
            "recommendations": self.recommendations_output,
            "tasks": self.tasks_output,
        }

    # ---------------------------------------- Legacy functions, for backward compatibility ----------------------------------------
    # ------------------------------ These functions are not used in the current version of the code. ------------------------------

    def analysis_insights(self, user_input: str) -> str:
        self.logger.warning(
            "The `analysis_insights` method is deprecated and will be removed in a future version. Use the `insights` method instead."
        )
        self.analysis_output = self.analysis(
            user_input=user_input,
            analysis_context="",
        )
        return self.insights(user_input)

    def visualizations(
        self, user_input: str, dir_path: Path = None
    ) -> list[Image.Image]:
        self.logger.warning(
            "The `visualizations` method is deprecated and will be removed in a future version. Use the `visualisation` method instead."
        )
        return [
            Image.open(
                self.visualisation(
                    user_input,
                    plot_path=(
                        str(dir_path / "plot.png") if dir_path is not None else None
                    ),
                )
            )
        ]

    def dataset_description(self, context: str = None) -> str:
        self.logger.warning(
            "The `dataset_description` method is deprecated and will be removed in a future version."
        )
        context = context or self.context
        df_desc_dict = {}
        for name, df in self.df_dict.items():
            self._gen_model.set_messages(
                messages=[
                    {
                        "content": Prompt("dataset_description_context_pt")
                        .format(context=context)
                        .text,
                        "role": "system",
                    },
                    {
                        "content": Prompt("dataset_description_pt")
                        .format(
                            df_head=df.head(),
                        )
                        .text,
                        "role": "user",
                    },
                ]
            )
            self._gen_model_kwargs = set_model_params(
                {
                    "temperature": 1,
                    "top_p": 0.3,
                    "frequency_penalty": 0.7,
                    "presence_penalty": 0.3,
                },
                self._gen_model_kwargs,
            )
            df_desc_dict[name] = (
                self._gen_model.run(**self._gen_model_kwargs).choices[0].message.content
            )
        self.dataset_description_output = "\n".join(
            [f"{name}:\n{desc}" for name, desc in df_desc_dict.items()]
        )
        return self.dataset_description_output

    def ai_queries_df(
        self, dataset_description: Optional[str] = None, context: Optional[str] = None
    ) -> str:
        self.logger.warning(
            "The `ai_queries_df` method is deprecated and will be removed in a future version."
        )
        context = context or self.context
        self.dataset_description_output = (
            dataset_description or self.dataset_description_output
        )
        if self.dataset_description_output is None:
            self.dataset_description_output = self.dataset_description(context)

        self._gen_model.set_messages(
            messages=[
                {
                    "content": Prompt("ai_queries_context_pt")
                    .format(context=context)
                    .text,
                    "role": "system",
                },
                {
                    "content": Prompt("ai_queries_pt")
                    .format(
                        df_details=format_df_with_describe(self.df_dict),
                    )
                    .text,
                    "role": "user",
                },
            ]
        )
        self._gen_model_kwargs = set_model_params(
            {
                "temperature": 1,
                "top_p": 0.3,
                "frequency_penalty": 0.7,
                "presence_penalty": 0.3,
            },
            self._gen_model_kwargs,
        )
        self.ai_queries = (
            self._gen_model.run(**self._gen_model_kwargs).choices[0].message.content
        )

        return self.ai_queries

    def analysis_recommendation(
        self,
        user_input: Optional[str] = None,
        number_of_recommendations: Optional[int] = 4,
    ):
        self.logger.warning(
            "The `analysis_recommendation` method is deprecated and will be removed in a future version."
        )
        formatted_user_input: str = (
            Prompt("format_user_input_pt").format(user_input=user_input)
            if user_input is not None
            else ""
        )

        self._gen_model.set_messages(
            messages=[
                {
                    "prompt": Prompt("analysis_recommendations_pt").format(
                        number_of_recommendations=number_of_recommendations,
                        df_details=format_df_with_info(self.df_dict),
                        formatted_user_input=formatted_user_input,
                    ),
                    "role": "system",
                },
            ]
        )
        self._gen_model_kwargs = set_model_params(
            {"temperature": 0.2}, self._gen_model_kwargs
        )
        recommendations = (
            self._gen_model.run(**self._gen_model_kwargs).choices[0].message.content
        )
        return recommendations
