# standard library imports
import os
import json
import time
import warnings
from typing import Union, Literal, Optional, Any

# third-party imports
import numpy as np
import pandas as pd
from pydantic import TypeAdapter

# local imports
from lyzr.data_analyzr.utils import (
    logging_decorator,
    deterministic_uuid,
    format_df_details,
    format_df_details,
    get_info_dict_from_df_dict,
    get_context_dict,
)
from lyzr.data_analyzr.models import (
    SupportedDBs,
    AnalysisTypes,
    VectorStoreConfig,
    DataConfig,
    OutputTypes,
    ParamsDict,
)
from lyzr.base.logger import set_logger
from lyzr.base.prompt import LyzrPromptFactory
from lyzr.base.errors import MissingValueError
from lyzr.base.llm import LyzrLLMFactory, LiteLLM
from lyzr.data_analyzr.plot_handler import PlotFactory
from lyzr.data_analyzr.file_utils import get_db_details
from lyzr.data_analyzr.analysis_handler import TxttoSQLFactory, PythonicAnalysisFactory

# imports for legacy usage
from PIL import Image
from pathlib import Path
from lyzr.base.llms import LLM
from pandas.errors import EmptyDataError


class DataAnalyzr:

    def __init__(
        self,
        analysis_type: Optional[Literal["sql", "ml", "skip"]] = None,
        api_key: Optional[str] = None,
        max_retries: Optional[int] = None,
        time_limit: Optional[int] = None,
        generator_llm: Optional[LiteLLM] = None,
        analysis_llm: Optional[LiteLLM] = None,
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
        self.params = ParamsDict(
            max_retries=3 if max_retries is None else max_retries,
            time_limit=30 if time_limit is None else time_limit,
            auto_train=True,
        )
        self.logger = set_logger(
            name="data_analyzr",
            logfilename=log_filename,
            log_level=log_level,
            print_log=print_log,
        )

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
            if generator_llm is None:
                self.generator_llm = LyzrLLMFactory.from_defaults(
                    model="gpt-4-1106-preview", api_key=api_key, seed=seed
                )
            elif isinstance(generator_llm, LiteLLM):
                self.generator_llm = generator_llm
            self.generator_llm.additional_kwargs["logger"] = self.logger
            if analysis_llm is None:
                self.analysis_llm = LyzrLLMFactory.from_defaults(
                    model="gpt-3.5-turbo", api_key=api_key, seed=seed
                )
            elif isinstance(analysis_llm, LiteLLM):
                self.analysis_llm = analysis_llm
            self.analysis_llm.additional_kwargs["logger"] = self.logger

            self.analysis_type = AnalysisTypes(analysis_type.lower().strip())
            self.context = "" if context is None else context.strip()
        (
            self.steps,
            self.code,
            self.analysis_guide,
            self.analysis_output,
            self.vector_store,
            self.dataset_description_output,
            self.ai_queries_output,
            self.visualisation_output,
            self.insights_output,
            self.recommendations_output,
            self.tasks_output,
        ) = (None,) * 11

        self.analysis = logging_decorator(logger=self.logger)(self.analysis)
        self.visualisation = logging_decorator(logger=self.logger)(self.visualisation)
        self.insights = logging_decorator(logger=self.logger)(self.insights)
        self.recommendations = logging_decorator(logger=self.logger)(
            self.recommendations
        )
        self.tasks = logging_decorator(logger=self.logger)(self.tasks)

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
            "The `df` parameter is deprecated and will be removed in a future version. Please use the `get_data` method to load data.",
            category=DeprecationWarning,
        )
        for param in ["model", "model_type", "model_name", "seed"]:
            if locals()[param] is not None:
                warnings.warn(
                    f"The `{param}` parameter is deprecated and will be removed in a future version. Please use the `analysis_model` parameter to set the analysis model, and the `generator_model` parameter to set the generation model.",
                    category=DeprecationWarning,
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
        self.generator_llm = LyzrLLMFactory.from_defaults(
            api_key=api_key,
            api_type=model_type,
            model=model_name or "gpt-4-1106-preview",
            seed=seed,
            **model_kwargs,
        )
        self.analysis_llm = LyzrLLMFactory.from_defaults(
            api_key=api_key,
            api_type=model_type,
            model=model_name or "gpt-3.5-turbo",
            seed=seed,
            **model_kwargs,
        )
        self.context = ""
        self.analysis_type = AnalysisTypes("ml")
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
                db_scope=AnalysisTypes("ml"),
                db_type=SupportedDBs("files"),
                db_config=TypeAdapter(DataConfig).validate_python(
                    {"datasets": {"Dataset": df}}
                ),
                vector_store_config=VectorStoreConfig(**{}),
                logger=self.logger,
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

    # Function to get datasets
    def get_data(
        self,
        db_type: Literal["files", "redshift", "postgres", "sqlite"],
        data_config: dict,
        vector_store_config: dict = {},
    ) -> Any:
        if not isinstance(data_config, dict):
            raise ValueError("data_config must be a dictionary.")
        data_config["db_type"] = SupportedDBs(db_type.lower().strip())
        self.database_connector, self.df_dict, self.vector_store = get_db_details(
            db_scope=self.analysis_type,
            db_type=data_config["db_type"],
            db_config=TypeAdapter(DataConfig).validate_python(data_config),
            vector_store_config=VectorStoreConfig(**vector_store_config),
            logger=self.logger,
        )

    def analysis(
        self,
        user_input: str,
        analysis_context: str,
    ):
        if self.analysis_type is AnalysisTypes.skip:
            if self.df_dict is None:
                self.logger.info(
                    "No analysis performed. Fetching dataframes from database."
                )
                self.df_dict = self.database_connector.fetch_dataframes_dict()
            self.analysis_output = self.df_dict
            self.analysis_guide = (
                "No analysis performed. Analysis output is the given dataframe."
            )
            self.analyzer = None
            return self.analysis_output
        if self.analysis_type is AnalysisTypes.sql:
            return self._txt_to_sql_analysis(
                self.analysis_llm, user_input, analysis_context
            )
        if self.analysis_type is AnalysisTypes.ml:
            return self._ml_analysis(
                self.analysis_llm,
                user_input,
                analysis_context,
            )

    def _ml_analysis(
        self,
        analysis_llm,
        user_input: str,
        analysis_context: str = None,
    ):
        self.analyzer = PythonicAnalysisFactory(
            llm=analysis_llm,
            df_dict=self.df_dict,
            logger=self.logger,
            context=analysis_context,
            vector_store=self.vector_store,
        )
        self.analysis_output = self.analyzer.run_analysis(
            user_input,
            max_retries=self.params.max_retries,
            time_limit=self.params.time_limit,
        )
        self.analysis_guide = self.analyzer.analysis_guide
        return self.analysis_output

    def _txt_to_sql_analysis(
        self, analysis_llm, user_input: str, analysis_context: str = None
    ):
        self.analyzer = TxttoSQLFactory(
            llm=analysis_llm,
            db_connector=self.database_connector,
            logger=self.logger,
            context=analysis_context,
            vector_store=self.vector_store,
        )
        self.analysis_output = self.analyzer.run_analysis(
            user_input,
            max_retries=self.params.max_retries,
            time_limit=self.params.time_limit,
        )
        self.analysis_guide = self.analyzer.code
        return self.analysis_output

    def visualisation(
        self,
        user_input: str,
        plot_context: str = None,
        plot_path: str = None,
    ):
        if plot_path is None:
            plot_path = Path(
                f"generated_plots/{deterministic_uuid([self.analysis_type.value, user_input])}.png"
            ).as_posix()
        else:
            plot_path = Path(plot_path).as_posix()
        self.visualisation_output = None

        self.plotter = PlotFactory(
            llm=self.analysis_llm,
            logger=self.logger,
            context=plot_context,
            plot_path=plot_path,
            df_dict={**self.df_dict, "analysis_output": self.analysis_output},
            analyzer=self.analyzer,
            analysis_output=self.analysis_output,
            vector_store=self.vector_store,
        )
        self.visualisation_output = self.plotter.get_visualisation(
            user_input,
            max_retries=self.params.max_retries,
            time_limit=self.params.time_limit,
        )
        return self.visualisation_output

    def insights(
        self,
        user_input: str,
        insights_context: Optional[str] = None,
        n_insights: Optional[int] = 3,
    ) -> str:
        if "analysis_guide" not in self.__dict__:
            self.analysis_guide = ""
        self.insights_output = self.generator_llm.run(
            messages=[
                LyzrPromptFactory(name="insights", prompt_type="system").get_message(
                    context=insights_context.strip(), n_insights=n_insights
                ),
                LyzrPromptFactory(name="insights", prompt_type="user").get_message(
                    user_input=user_input,
                    analysis_guide=self.analysis_guide,
                    analysis_output=format_df_details(
                        output_df=(
                            self.analysis_output
                            if self.analysis_output is not None
                            else self.df_dict
                        )
                    ),
                    date=time.strftime("%d %b %Y"),
                ),
            ],
            temperature=0.3,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            logger=self.logger,
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
                "The `insights` parameter is deprecated and will be removed in a future version. Please use the `insights_output` attribute to set the insights.",
                category=DeprecationWarning,
            )
            self.insights_output = insights
        if schema is not None:
            warnings.warn(
                "The `schema` parameter is deprecated and will be removed in a future version. Please use the `recs_format` parameter to set the recommendations format as a dictionary.",
                category=DeprecationWarning,
            )
            for elem in schema:
                if isinstance(elem, dict):
                    recs_format = elem
                    break

        system_message_sections = ["context"]
        system_message_dict = {}
        df_details = format_df_details(
            output_df=(
                self.analysis_output
                if self.analysis_output is not None
                else self.df_dict
            )
        )
        user_message_dict = {
            "user_input": user_input,
            "analysis_output": f"Analysis output:\n{df_details}",
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
        self.recommendations_output = self.generator_llm.run(
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
            logger=self.logger,
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
                "The `insights` parameter is deprecated and will be removed in a future version. Please use the `insights_output` attribute to set the insights.",
                category=DeprecationWarning,
            )
            self.insights_output = insights
        if recommendations is not None:
            warnings.warn(
                "The `recommendations` parameter is deprecated and will be removed in a future version. Please use the `recommendations_output` attribute to set the recommendations.",
                category=DeprecationWarning,
            )
            self.recommendations_output = recommendations
        self.tasks_output = self.generator_llm.run(
            messages=[
                LyzrPromptFactory(name="tasks", prompt_type="system").get_message(
                    context=tasks_context, n_tasks=n_tasks
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
            logger=self.logger,
        ).message.content.strip()
        return self.tasks_output

    def ai_queries(
        self,
        context: Optional[str] = None,
    ) -> str:
        context = self.context if context is None else context
        context = context.strip() + "\n\n" if context.strip() != "" else ""
        schema = {
            "query1": "string",
            "query2": "string",
            "query3": "string",
        }
        messages = [
            LyzrPromptFactory(name="ai_queries", prompt_type="system").get_message(
                context=context,
                schema=schema,
            ),
            LyzrPromptFactory(name="ai_queries", prompt_type="user").get_message(
                df_details=format_df_details(self.df_dict)
            ),
        ]
        ai_queries_output = self.generator_llm.run(
            messages=messages,
            temperature=1,
            top_p=0.3,
            frequency_penalty=0.7,
            presence_penalty=0.3,
            response_format={"type": "json_object"},
        ).message.content
        self.ai_queries_output = json.loads(ai_queries_output)

        return self.ai_queries_output

    def ask(
        self,
        user_input: str = None,
        outputs: list[
            OutputTypes.visualisation.value,
            OutputTypes.insights.value,
            OutputTypes.recommendations.value,
            OutputTypes.tasks.value,
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
        if outputs is None or len(outputs) == 0:
            outputs = [
                OutputTypes.visualisation,
                OutputTypes.insights,
                OutputTypes.recommendations,
                OutputTypes.tasks,
            ]
        else:
            outputs = [OutputTypes._member_map_[output] for output in outputs]
        if user_input is None and self.user_input is None:
            raise MissingValueError(
                "`user_input` is a required parameter to generate outputs."
            )
        if user_input is None:
            user_input = self.user_input
        context = get_context_dict(context_str=self.context, context_dict=context)
        counts = counts or {}

        if self.analysis_output is None or rerun_analysis:
            self.analysis_output = self.analysis(
                user_input=user_input,
                analysis_context=context.get("analysis"),
            )

        if OutputTypes.visualisation in outputs:
            self.visualisation_output = self.visualisation(
                user_input=user_input,
                plot_context=context.get("visualisation"),
                plot_path=plot_path,
            )
        else:
            self.visualisation_output = ""

        if (
            OutputTypes.insights in outputs
            or OutputTypes.tasks in outputs
            or (OutputTypes.recommendations in outputs and use_insights)
        ):
            self.insights_output = self.insights(
                user_input=user_input,
                insights_context=context.get("insights"),
                n_insights=counts.get("insights", 3),
            )
        else:
            self.insights_output = ""

        if OutputTypes.recommendations in outputs or OutputTypes.tasks in outputs:
            self.recommendations_output = self.recommendations(
                user_input=user_input,
                use_insights=use_insights,
                recs_format=recs_format,
                recommendations_context=context.get("recommendations"),
                n_recommendations=counts.get("recommendations", 3),
                output_type=recs_output_type or "text",
            )
        else:
            self.recommendations_output = ""

        if OutputTypes.tasks in outputs:
            self.tasks_output = self.tasks(
                user_input=user_input,
                tasks_context=context.get("tasks"),
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
            "The `analysis_insights` method is deprecated and will be removed in a future version. Use the `insights` method instead.",
            category=DeprecationWarning,
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
            "The `visualizations` method is deprecated and will be removed in a future version. Use the `visualisation` method instead.",
            category=DeprecationWarning,
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
            "The `dataset_description` method is deprecated and will be removed in a future version.",
            category=DeprecationWarning,
        )
        context = context or self.context
        df_desc_dict = {}
        for name, df in self.df_dict.items():
            df_desc_dict[name] = self.generator_llm.run(
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
            "The `ai_queries_df` method is deprecated and will be removed in a future version. Use the `ai_queries` method instead.",
            category=DeprecationWarning,
        )
        context = context or self.context
        self.dataset_description_output = (
            dataset_description or self.dataset_description_output
        )
        if self.dataset_description_output is None:
            self.dataset_description_output = self.dataset_description(context)

        self.ai_queries_output = self.generator_llm.run(
            messages=[
                LyzrPromptFactory(name="ai_queries", prompt_type="system").get_message(
                    use_sections=["context", "external_context", "task"],
                    context=context.strip() + "\n\n",
                ),
                LyzrPromptFactory(name="ai_queries", prompt_type="user").get_message(
                    df_details=format_df_details(self.df_dict)
                ),
            ],
            temperature=1,
            top_p=0.3,
            frequency_penalty=0.7,
            presence_penalty=0.3,
        ).message.content.strip()

        return self.ai_queries_output

    def analysis_recommendation(
        self,
        user_input: Optional[str] = None,
        number_of_recommendations: Optional[int] = 4,
    ):
        warnings.warn(
            "The `analysis_recommendation` method is deprecated and will be removed in a future version.",
            category=DeprecationWarning,
        )
        formatted_user_input: str = (
            LyzrPromptFactory("format_user_input", "system")
            .get_message(user_input=user_input)
            .content
            if user_input is not None
            else ""
        )
        recommendations = self.generator_llm.run(
            messages=[
                LyzrPromptFactory("analysis_recommendations", "system").get_message(
                    number_of_recommendations=number_of_recommendations,
                    df_details=format_df_details(self.df_dict),
                    formatted_user_input=formatted_user_input,
                ),
            ],
            temperature=0.2,
        ).message.content.strip()
        return recommendations
