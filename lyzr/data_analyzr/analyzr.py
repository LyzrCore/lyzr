# standard library imports
import json
import time
from pathlib import Path
from typing import Literal, Optional, Any

# third-party imports
import pandas as pd

# local imports
from lyzr.data_analyzr.models import (
    AnalysisTypes,
    OutputTypes,
    ParamsDict,
)
from lyzr.base.prompt import LyzrPromptFactory
from lyzr.base.llm import LyzrLLMFactory, LiteLLM
from lyzr.data_analyzr.db_models import SupportedDBs, VectorStoreConfig, DataConfig


class DataAnalyzr:

    def __init__(
        self,
        analysis_type: Literal["sql", "ml", "skip"],
        api_key: Optional[str] = None,
        class_params: Optional[dict] = None,
        generator_llm: Optional[LiteLLM] = None,
        analysis_llm: Optional[LiteLLM] = None,
        context: Optional[str] = None,
        log_params: Optional[dict] = None,
    ):
        try:
            self.analysis_type = AnalysisTypes(analysis_type.lower().strip())
        except ValueError:
            raise ValueError(
                f"Invalid value for `analysis_type`. Must be one of {', '.join([at.value for at in AnalysisTypes])}."
            )
        class_params = class_params or {}
        self.params = ParamsDict(
            max_retries=class_params.get("max_retries", None),
            time_limit=class_params.get("time_limit", None),
            auto_train=class_params.get("auto_train", None),
        )
        from lyzr.base.logger import set_logger

        log_params = log_params or {}
        self.logger = set_logger(
            name="data_analyzr",
            logfilename=log_params.get("log_filename", "dataanalyzr.csv"),
            log_level=log_params.get(
                "log_level", "INFO"
            ),  # one of "NOTSET", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
            print_log=log_params.get("print_log", False),
        )
        if isinstance(generator_llm, LiteLLM):
            self.generator_llm = generator_llm
        else:
            self.generator_llm = LyzrLLMFactory.from_defaults(
                model="gpt-4o", api_key=api_key, seed=0
            )

        self.generator_llm.additional_kwargs["logger"] = self.logger
        if isinstance(analysis_llm, LiteLLM):
            self.analysis_llm = analysis_llm
        else:
            self.analysis_llm = LyzrLLMFactory.from_defaults(
                model="gpt-4o", api_key=api_key, seed=0
            )
        self.analysis_llm.additional_kwargs["logger"] = self.logger

        self.context = "" if context is None else context.strip()
        (
            self.df_dict,
            self.database_connector,
            self.vector_store,
            self.analyser,
            self.analysis_code,
            self.analysis_guide,
            self.analysis_output,
            self.plotter,
            self.plot_code,
            self.plot_output,
            self.insights_output,
            self.recommendations_output,
            self.tasks_output,
            self.ai_queries_output,
        ) = (None,) * 14

        from lyzr.data_analyzr.utils import logging_decorator

        self.analysis = logging_decorator(logger=self.logger)(self.analysis)
        self.visualisation = logging_decorator(logger=self.logger)(self.visualisation)
        self.insights = logging_decorator(logger=self.logger)(self.insights)
        self.recommendations = logging_decorator(logger=self.logger)(
            self.recommendations
        )
        self.tasks = logging_decorator(logger=self.logger)(self.tasks)

    def get_data(
        self,
        db_type: Literal["files", "redshift", "postgres", "sqlite"],
        data_config: dict,
        vector_store_config: dict = {},
    ) -> Any:
        from pydantic import TypeAdapter
        from lyzr.data_analyzr.file_utils import get_db_details

        if not isinstance(data_config, dict):
            raise ValueError("data_config must be a dictionary.")
        data_config["db_type"] = SupportedDBs(db_type.lower().strip())
        self.database_connector, self.df_dict, self.vector_store = get_db_details(
            analysis_type=self.analysis_type,
            db_type=data_config["db_type"],
            db_config=TypeAdapter(DataConfig).validate_python(data_config),
            vector_store_config=VectorStoreConfig(**vector_store_config),
            logger=self.logger,
        )

    def analysis(
        self,
        user_input: str,
        analysis_context: str,
        time_limit: int,
        max_retries: int,
        auto_train: bool,
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
            self.analyser = None
            return self.analysis_output

        from lyzr.data_analyzr.analysis_handler import (
            PythonicAnalysisFactory,
            TxttoSQLFactory,
        )

        analyser_args = dict(
            llm=self.analysis_llm,
            logger=self.logger,
            context=analysis_context,
            vector_store=self.vector_store,
            time_limit=time_limit,
            max_retries=max_retries,
            auto_train=auto_train,
        )
        if self.analysis_type is AnalysisTypes.sql:
            self.analyser = TxttoSQLFactory(
                **analyser_args,
                db_connector=self.database_connector,
            )
        if self.analysis_type is AnalysisTypes.ml:
            self.analyser = PythonicAnalysisFactory(
                **analyser_args,
                df_dict=self.df_dict,
            )
        self.analysis_output = self.analyser.run_analysis(user_input)
        self.analysis_guide = self.analyser.guide
        self.analysis_code = self.analyser.code
        return self.analysis_output

    def visualisation(
        self,
        user_input: str,
        plot_context: str,
        time_limit: int,
        max_retries: int,
        auto_train: bool,
        plot_path: str = None,
    ):
        from lyzr.data_analyzr.utils import deterministic_uuid
        from lyzr.data_analyzr.analysis_handler import PlotFactory

        if plot_path is None:
            plot_path = Path(
                f"generated_plots/{deterministic_uuid([self.analysis_type.value, user_input])}.png"
            ).as_posix()
        else:
            plot_path = Path(plot_path).as_posix()
        if self.df_dict is None:
            data_kwargs = {"connector": self.database_connector}
        else:
            data_kwargs = {"df_dict": self.df_dict}
        if isinstance(self.analysis_output, pd.DataFrame):
            data_kwargs["analysis_output"] = {"analysis_output": self.analysis_output}
        elif isinstance(self.analysis_output, dict):
            data_kwargs["analysis_output"] = self.analysis_output
        self.plot_output = None
        self.plotter = PlotFactory(
            llm=self.analysis_llm,
            logger=self.logger,
            context=plot_context,
            plot_path=plot_path,
            data_kwargs=data_kwargs,
            vector_store=self.vector_store,
            time_limit=time_limit,
            max_retries=max_retries,
            auto_train=auto_train,
        )
        self.plot_output = self.plotter.get_visualisation(user_input=user_input)
        self.plot_code = self.plotter.code
        return self.plot_output

    def insights(
        self,
        user_input: str,
        insights_context: Optional[str] = None,
        n_insights: Optional[int] = 3,
    ) -> str:
        from lyzr.data_analyzr.utils import format_df_details

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
        user_input: Optional[str],
        from_insights: Optional[bool] = True,
        recs_format: Optional[dict] = None,
        recommendations_context: Optional[str] = None,
        n_recommendations: Optional[int] = 3,
        output_type: Optional[Literal["text", "json"]] = "text",
    ) -> str:
        from lyzr.data_analyzr.utils import format_df_details

        system_message_sections = ["context"]
        system_message_dict = {}
        llm_params = dict(
            temperature=0.3,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            logger=self.logger,
        )
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

        if from_insights:
            system_message_sections.append("task_with_insights")
            user_message_dict["insights"] = (
                f"\nAnalysis insights:\n{self.insights_output}\n"
            )
        else:
            system_message_sections.append("task_no_insights")
            user_message_dict["insights"] = ""

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
            llm_params["response_format"] = {"type": "json_object"}
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
            **llm_params,
        ).message.content
        return self.recommendations_output

    def tasks(
        self,
        user_input: Optional[str] = None,
        tasks_context: Optional[str] = None,
        n_tasks: Optional[int] = 5,
    ) -> str:
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
        recommendations_params: Optional[dict] = None,
        counts: dict = None,  # keys: insights, recommendations, tasks
        context: dict = None,  # keys: analysis, insights, recommendations, tasks
        **kwargs,  # rerun_analysis, max_retries, time_limit, auto_train
    ):
        rerun_analysis = (
            kwargs.get("rerun_analysis")
            if isinstance(kwargs.get("rerun_analysis"), bool)
            else True
        )
        if outputs is None or len(outputs) == 0:
            outputs = [
                OutputTypes.visualisation,
                OutputTypes.insights,
                OutputTypes.recommendations,
                OutputTypes.tasks,
            ]
        else:
            outputs = [OutputTypes._member_map_[output] for output in outputs]
        assert (
            user_input is not None
            and isinstance(user_input, str)
            and user_input.strip() != ""
        ), "user_input is a required string parameter to generate outputs."
        user_input = user_input.strip()
        from lyzr.data_analyzr.utils import get_context_dict

        context = get_context_dict(context_str=self.context, context_dict=context)
        counts = counts or {}
        recommendations_params = recommendations_params or {}

        time_limit = kwargs.get("time_limit", self.params.time_limit)
        max_retries = kwargs.get("max_retries", self.params.max_retries)
        auto_train = kwargs.get("auto_train", self.params.auto_train)

        if self.analysis_output is None or rerun_analysis:
            self.analysis_output = self.analysis(
                user_input=user_input,
                analysis_context=context.get("analysis"),
                time_limit=time_limit,
                max_retries=max_retries,
                auto_train=auto_train,
            )

        if OutputTypes.visualisation in outputs:
            self.plot_output = self.visualisation(
                user_input=user_input,
                plot_context=context.get("visualisation"),
                plot_path=plot_path,
                time_limit=time_limit,
                max_retries=max_retries,
                auto_train=auto_train,
            )
        else:
            self.plot_output = ""
        if (
            OutputTypes.insights in outputs
            or OutputTypes.tasks in outputs
            or (
                OutputTypes.recommendations in outputs
                and recommendations_params.get("from_insights", True)
            )
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
                from_insights=recommendations_params.get("from_insights", True),
                recs_format=recommendations_params.get("json_format", None),
                recommendations_context=context.get("recommendations"),
                n_recommendations=counts.get("recommendations", 3),
                output_type=recommendations_params.get("output_type", "text"),
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
            "visualisation": self.plot_output,
            "insights": self.insights_output,
            "recommendations": self.recommendations_output,
            "tasks": self.tasks_output,
        }

    def ai_queries(
        self,
        context: Optional[str] = None,
    ) -> str:
        from lyzr.data_analyzr.utils import format_df_details

        context = self.context if context is None else context
        context = context.strip() + "\n\n" if context.strip() != "" else ""
        schema = {
            "type_of_analysis1": ["query1", "query2", "query3", "query4"],
            "type_of_analysis2": ["query1", "query2", "query3", "query4"],
            "type_of_analysis3": ["query1", "query2", "query3", "query4"],
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
