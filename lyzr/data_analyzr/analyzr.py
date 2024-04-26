# standard library imports
import os
import time
import uuid
import warnings
from typing import Union, Literal, Optional, Any

# third-party imports
import numpy as np
import pandas as pd

# local imports
from lyzr.base.prompt import LyzrPromptFactory
from lyzr.data_analyzr.file_utils import get_db_details
from lyzr.data_analyzr.txt_to_sql_utils import TxttoSQLFactory
from lyzr.data_analyzr.ml_analysis_utils import MLAnalysisFactory
from lyzr.base.errors import (
    MissingValueError,
)
from lyzr.base.llm import (
    LyzrLLMFactory,
    LiteLLM,
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
from lyzr.base.llms import LLM
from pandas.errors import EmptyDataError

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")


class DataAnalyzr:

    def __init__(
        self,
        analysis_type: Optional[Literal["sql", "ml", "skip"]] = None,
        api_key: Optional[str] = None,
        max_retries: Optional[int] = 5,
        generator_model: Optional[LiteLLM] = None,
        analysis_model: Optional[LiteLLM] = None,
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
        seed: Optional[int] = 0,
    ):
        api_key = (
            api_key or os.environ.get("API_KEY") or os.environ.get("OPENAI_API_KEY")
        )
        if api_key is None:
            raise MissingValueError(
                "Provide a value for `api_key` or set the `OPENAI_API_KEY` environment variable."
            )
        self.max_retries = max_retries
        self.user_input = user_input
        self.log_filename = log_filename
        self._set_logger(log_level, print_log)

        if df is not None:
            # legacy usage
            self._legacy_usage(
                api_key=api_key,
                df=df,
                user_input=user_input,
                model=model,
                model_type=model_type,
                model_name=model_name,
                seed=seed,
            )
        else:
            if analysis_type is None:
                raise MissingValueError("`analysis_type` is a required parameter.")
            if generator_model is None:
                self.generator_model = LyzrLLMFactory.from_defaults(
                    model="gpt-4-1106-preview", api_key=api_key, seed=seed
                )
            elif isinstance(generator_model, LiteLLM):
                self.generator_model = generator_model
            self.generator_model.additional_kwargs["logger"] = self.logger
            if analysis_model is None:
                self.analysis_model = LyzrLLMFactory.from_defaults(
                    model="gpt-3.5-turbo", api_key=api_key, seed=seed
                )
            elif isinstance(analysis_model, LiteLLM):
                self.analysis_model = analysis_model
            self.analysis_model.additional_kwargs["logger"] = self.logger

            self.analysis_type = analysis_type.lower().strip()
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
        api_key: str,
        df: Union[str, pd.DataFrame],
        user_input: Optional[str],
        model: Optional[LLM],
        model_type: Optional[str],
        model_name: Optional[str],
        seed: int,
    ):
        warnings.warn(
            "The `df` parameter is deprecated and will be removed in a future version. Please use the `get_data` method to load data."
        )
        for param in ["model", "model_type", "model_name", "seed"]:
            if locals()[param] is not None:
                warnings.warn(
                    f"The `{param}` parameter is deprecated and will be removed in a future version. Please use the `analysis_model` parameter to set the analysis model, and the `generator_model` parameter to set the generation model."
                )
        if isinstance(model, LLM):
            api_key = model.api_key
            model_name = model.model_name
            model_type = model.model_type
            model_kwargs = {}
            for attr in model.__dict__:
                if attr not in ["api_key", "model_name"]:
                    model_kwargs[attr] = model.__dict__[attr]
        else:
            model_kwargs = {}
            model_type = model_type or os.environ.get("MODEL_TYPE", None)
            model_name = model_name or os.environ.get("MODEL_NAME", None)
        self.generator_model = LyzrLLMFactory.from_defaults(
            api_key=api_key,
            api_type=model_type,
            model=model_name or "gpt-4-1106-preview",
            seed=seed,
            **model_kwargs,
        )
        self.analysis_model = LyzrLLMFactory.from_defaults(
            api_key=api_key,
            api_type=model_type,
            model=model_name or "gpt-3.5-turbo",
            seed=seed,
            **model_kwargs,
        )
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
            for handler in self.logger.handlers:
                try:
                    handler.close()
                except Exception:
                    pass
            self.logger.handlers.clear()

        if print_log:
            # output logs to stdout
            handler = logging.StreamHandler(sys.stdout)
            handler.setLevel(numeric_level)
            self.logger.addHandler(handler)

        log_filename = self.log_filename
        dir_path = os.path.dirname(log_filename)
        if dir_path.strip() != "":
            os.makedirs(dir_path, exist_ok=True)
        file_handler = logging.FileHandler(
            log_filename, mode="a"
        )  # Open the log file in append mode
        file_handler.setLevel(numeric_level)

        # Optionally, you can set a formatter for the file handler if you want a different format for file logs
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s\n%(message)s\n",
            datefmt="%d-%b-%y %H:%M:%S %Z",
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
        if self.analysis_type == "sql" and analysis_steps is None:
            return self._txt_to_sql_analysis(
                self.analysis_model, user_input, analysis_context
            )
        if self.analysis_type == "ml" or analysis_steps is not None:
            return self._ml_analysis(
                self.analysis_model,
                user_input,
                analysis_context,
                analysis_steps=analysis_steps,
            )

    def _ml_analysis(
        self,
        analysis_model,
        user_input: str,
        analysis_context: str = None,
        analysis_steps: dict = None,
    ):
        self.analyzer = MLAnalysisFactory(
            model=analysis_model,
            data_dict=self.df_dict,
            data_info_dict=self.df_info_dict,
            logger=self.logger,
            context=analysis_context,
        )
        if analysis_steps is not None:
            _, data = self.analyzer.get_analysis_from_steps(analysis_steps)
            return data
        else:
            self.analysis_output = self.analyzer.run_complete_analysis(user_input)
            self.analysis_guide = self.analyzer.analysis_guide
            return self.analysis_output

    def _txt_to_sql_analysis(
        self, analysis_model, user_input: str, analysis_context: str = None
    ):
        self.analyzer = TxttoSQLFactory(
            model=analysis_model,
            db_connector=self.database_connector,
            logger=self.logger,
            context=analysis_context,
            vector_store=self.vector_store,
        )
        self.analysis_output = self.analyzer.run_complete_analysis(user_input)
        self.analysis_guide = self.analyzer.analysis_guide
        return self.analysis_output

    def visualisation(
        self,
        user_input: str,
        plot_context: str = None,
        plot_path: str = None,
    ):
        if plot_path is None:
            plot_path = Path(f"generated_plots/{str(uuid.uuid4())}.png").as_posix()
        else:
            plot_path = Path(plot_path).as_posix()
        self.visualisation_output = None
        # self.start_time = time.time()

        self.logger.info("Generating visualisation\n")
        plotter = PlotFactory(
            model=self.analysis_model,
            logger=self.logger,
            plot_context=plot_context,
            plot_path=plot_path,
            df_dict=self.df_dict,
            database_connector=self.database_connector,
            analysis_type=self.analysis_type,
            analyzer=self.analyzer,
            analysis_output=self.analysis_output,
        )
        self.visualisation_output = plotter.get_visualisation(user_input)
        return self.visualisation_output

    def insights(
        self,
        user_input: str,
        insights_context: Optional[str] = None,
        n_insights: Optional[int] = 3,
    ) -> str:
        if "analysis_guide" not in self.__dict__:
            self.analysis_guide = ""
        self.insights_output = self.generator_model.run(
            messages=[
                LyzrPromptFactory(name="insights", prompt_type="system").get_message(
                    context=insights_context.strip() + "\n\n", n_insights=n_insights
                ),
                LyzrPromptFactory(name="insights", prompt_type="user").get_message(
                    user_input=user_input,
                    analysis_guide=self.analysis_guide,
                    analysis_output=format_df_with_describe(self.analysis_output),
                    date=time.strftime("%d %b %Y"),
                ),
            ],
            temperature=0.3,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        ).message.content.strip()
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
            warnings.warn(
                "The `insights` parameter is deprecated and will be removed in a future version. Please use the `insights_output` attribute to set the insights."
            )
            self.insights_output = insights
        if schema is not None:
            warnings.warn(
                "The `schema` parameter is deprecated and will be removed in a future version. Please use the `recs_format` parameter to set the recommendations format as a dictionary."
            )
            for elem in schema:
                if isinstance(elem, dict):
                    recs_format = elem
                    break

        system_message_sections = ["context"]
        system_message_dict = {}
        user_message_dict = {
            "user_input": user_input,
            "analysis_output": f"Analysis output:\n{format_df_with_describe(self.analysis_output)}",
        }
        if recommendations_context is not None and recommendations_context != "":
            system_message_sections.append("external_context")
            system_message_dict["context"] = recommendations_context.strip() + "\n\n"

        if not use_insights:
            insights = None
            system_message_sections.append("task_no_insights")
            user_message_dict["insights"] = ""
        else:
            system_message_sections.append("task_with_insights")
            user_message_dict["insights"] = (
                f"\nAnalysis insights:\n{self.insights_output}\n"
            )

        if output_type.lower().strip() == "json":
            recs_format = recs_format or {
                "Recommendation": "string",
                "Basis of the Recommendation": "string",
                "Impact if implemented": "string",
            }
            system_message_sections.append("json_type")
            system_message_dict["output_json_keys"] = ", ".join(recs_format.keys())
            system_message_dict["json_schema"] = [
                recs_format for _ in range(n_recommendations)
            ]
        elif output_type.lower().strip() == "text":
            system_message_sections.append("text_type")
        system_message_dict["n_recommendations"] = n_recommendations

        system_message_sections.append("closing")
        self.recommendations_output = self.generator_model.run(
            messages=[
                LyzrPromptFactory(
                    name="recommendations", prompt_type="system"
                ).get_message(
                    use_sections=system_message_sections, **system_message_dict
                ),
                LyzrPromptFactory(
                    name="recommendations", prompt_type="user"
                ).get_message(**user_message_dict),
            ],
            temperature=0.3,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            response_format=(
                {"type": "json_object"} if output_type == "json" else {"type": "text"}
            ),
        ).message.content
        return self.recommendations_output

    def tasks(
        self,
        user_input: Optional[str] = None,
        tasks_context: Optional[str] = None,
        n_tasks: Optional[int] = 5,
        # legacy usage
        insights: Optional[str] = None,
        recommendations: Optional[str] = None,
    ) -> str:
        if insights is not None:
            warnings.warn(
                "The `insights` parameter is deprecated and will be removed in a future version. Please use the `insights_output` attribute to set the insights."
            )
            self.insights_output = insights
        if recommendations is not None:
            warnings.warn(
                "The `recommendations` parameter is deprecated and will be removed in a future version. Please use the `recommendations_output` attribute to set the recommendations."
            )
            self.recommendations_output = recommendations
        self.tasks_output = self.generator_model.run(
            messages=[
                LyzrPromptFactory(name="tasks", prompt_type="system").get_message(
                    context=tasks_context.strip() + "\n\n", n_tasks=n_tasks
                ),
                LyzrPromptFactory(name="tasks", prompt_type="user").get_message(
                    user_input=user_input,
                    insights=self.insights_output,
                    recommendations=self.recommendations_output,
                ),
            ],
            temperature=1,
            top_p=0.3,
            frequency_penalty=0.7,
            presence_penalty=0.3,
        ).message.content.strip()
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
            raise MissingValueError(
                "`user_input` is a required parameter to generate outputs."
            )
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
        warnings.warn(
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
        warnings.warn(
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
        warnings.warn(
            "The `dataset_description` method is deprecated and will be removed in a future version."
        )
        context = context or self.context
        df_desc_dict = {}
        for name, df in self.df_dict.items():
            df_desc_dict[name] = self.generator_model.run(
                messages=[
                    LyzrPromptFactory(
                        name="dataset_description", prompt_type="system"
                    ).get_message(context=context.strip() + "\n\n"),
                    LyzrPromptFactory(
                        name="dataset_description", prompt_type="user"
                    ).get_message(
                        df_head=df.head(),
                        headers=df.columns.tolist(),
                    ),
                ],
                temperature=1,
                top_p=0.3,
                frequency_penalty=0.7,
                presence_penalty=0.3,
            ).message.content.strip()
        self.dataset_description_output = "\n".join(
            [f"{name}:\n{desc}" for name, desc in df_desc_dict.items()]
        )
        return self.dataset_description_output

    def ai_queries_df(
        self, dataset_description: Optional[str] = None, context: Optional[str] = None
    ) -> str:
        warnings.warn(
            "The `ai_queries_df` method is deprecated and will be removed in a future version."
        )
        context = context or self.context
        self.dataset_description_output = (
            dataset_description or self.dataset_description_output
        )
        if self.dataset_description_output is None:
            self.dataset_description_output = self.dataset_description(context)

        self.ai_queries = self.generator_model.run(
            messages=[
                LyzrPromptFactory(name="ai_queries", prompt_type="system").get_message(
                    context=context.strip() + "\n\n"
                ),
                LyzrPromptFactory(name="ai_queries", prompt_type="user").get_message(
                    df_details=format_df_with_describe(self.df_dict)
                ),
            ],
            temperature=1,
            top_p=0.3,
            frequency_penalty=0.7,
            presence_penalty=0.3,
        ).message.content.strip()

        return self.ai_queries

    def analysis_recommendation(
        self,
        user_input: Optional[str] = None,
        number_of_recommendations: Optional[int] = 4,
    ):
        warnings.warn(
            "The `analysis_recommendation` method is deprecated and will be removed in a future version."
        )
        formatted_user_input: str = (
            LyzrPromptFactory("format_user_input", "system")
            .get_message(user_input=user_input)
            .content
            if user_input is not None
            else ""
        )
        recommendations = self.generator_model.run(
            messages=[
                LyzrPromptFactory("analysis_recommendations", "system").get_message(
                    number_of_recommendations=number_of_recommendations,
                    df_details=format_df_with_info(self.df_dict),
                    formatted_user_input=formatted_user_input,
                ),
            ],
            temperature=0.2,
        ).message.content.strip()
        return recommendations
