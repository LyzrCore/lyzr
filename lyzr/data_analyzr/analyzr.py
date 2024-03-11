# standard library imports
import os
import io
import time
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

# imports for logging
import sys
import logging


class DataAnalyzr:

    def __init__(
        self,
        analysis_type: Literal["sql", "ml", "skip"],
        api_key: Optional[str] = None,
        gen_model: Optional[Union[dict, LLM]] = None,
        analysis_model: Optional[Union[dict, LLM]] = None,
        user_input: Optional[str] = None,
        context: Optional[str] = None,
        log_level: Optional[
            Literal["NOTSET", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        ] = "INFO",
        print_log: Optional[bool] = False,
        log_filename: Optional[str] = "dataanalyzr.log",
    ):
        self.api_key = (
            api_key or os.environ.get("API_KEY") or os.environ.get("OPENAI_API_KEY")
        )
        if self.api_key is None:
            raise MissingValueError("api key")

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

        self.context = context or ""
        self.user_input = user_input

        self.log_filename = log_filename
        self._set_logger(log_level, print_log)

        (
            self.dataset_description_output,
            self.ai_queries,
            self.analysis_output,
            self.insights_output,
            self.recommendations_output,
            self.tasks_output,
        ) = (None, None, None, None, None, None)

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
            # "bigquery",
            "snowflake",
            "mysql",
            "sqlite",
            # "azure_sql",
            # "adls",
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
        self.df_info_dict = None
        if self.df_dict is not None:
            self.df_info_dict = {}
            for name, df in self.df_dict.items():
                if not isinstance(df, pd.DataFrame):
                    continue
                buffer = io.StringIO()
                df.info(buf=buffer)
                self.df_info_dict[name] = buffer.getvalue()

        # if self.database_connector is not None:
        #     self.df_info_dict = (
        #         self.database_connector.fetch_dataframes_info_dict()
        #     )  # TODO: Implement this method in child classes of DatabaseConnector

    def dataset_description(self, context: str) -> str:
        """
        Generate a brief description of the dataset currently in use.

        Returns:
        - str: A string providing a description of the dataset.
        """
        if "dataset_description" in self.__dict__ and self.context == context:
            if self.dataset_description is not None:
                return self.dataset_description

        context = context or self.context
        self._gen_model.set_messages(
            messages=[
                {
                    "content": Prompt("dataset_description_context_pt")
                    .format(context=context)
                    .text,
                    "role": "system",
                },
                {
                    "content": Prompt("describe_dataset_pt")
                    .format(
                        df_head=(
                            self.df_info_dict
                            if self.df_dict is None
                            else _format_df_dict_head(self.df_dict)
                        ),
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
        self.dataset_description_output = (
            self._gen_model.run(**self._gen_model_kwargs).choices[0].message.content
        )
        return self.dataset_description_output

    def ai_queries_df(
        self, dataset_description: Optional[str] = None, context: Optional[str] = None
    ) -> str:
        """
        Returns AI-generated queries for data analysis related to the dataset.

        Parameters:
        - dataset_description (str, optional):
            A description of the dataset. If not provided, it will be generated.

        Returns:
        - str: Queries for data analysis related to the dataset.
        """
        if "ai_queries" in self.__dict__ and self.context == context:
            if self.ai_queries is not None:
                return self.ai_queries

        context = context or self.context

        if self.ai_queries is not None:
            return self.ai_queries

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
                        dataset_description=self.dataset_description_output,
                        df_head=(
                            self.df_info_dict
                            if self.df_dict is None
                            else _format_df_dict_head(self.df_dict)
                        ),
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

    def analysis(
        self,
        user_input: str,
        analysis_context: str,
    ):
        if self.analysis_type == "skip":
            if self.df_dict is None:
                self.logger.info(
                    "No analysis performed. Fetching dataframes from database."
                )
                self.df_dict = self.database_connector.fetch_dataframes_dict()
            self.analysis_output = self.df_dict
            self.analysis_steps = (
                "No analysis performed. Analysis output is the given dataframe."
            )
            return self.analysis_output
        self._analysis_model = self._analysis_model or {
            "type": "openai",
            "name": "gpt-3.5-turbo-1106",
            "kwargs": {},
        }
        if isinstance(self._analysis_model, LLM):
            self.analysis_model_kwargs = {}
        elif isinstance(self._analysis_model, dict):
            self.analysis_model_kwargs = self._analysis_model.get("kwargs", {})
            self._analysis_model = get_model(
                self.api_key,
                self._analysis_model.get("type"),
                self._analysis_model.get("name"),
                **self.analysis_model_kwargs,
            )
        if self.analysis_type == "sql":
            return self._txt_to_sql_analysis(user_input, analysis_context)
        if self.analysis_type == "ml":
            return self._ml_analysis(user_input, analysis_context)

    def _ml_analysis(self, user_input: str, analysis_context: str = None):
        self.analyzer = MLAnalysisFactory(
            model=self._analysis_model,
            data_dict=self.df_dict,
            data_info_dict=self.df_info_dict,
            logger=self.logger,
            context=analysis_context,
            model_kwargs=self.analysis_model_kwargs,
        )
        self.analysis_output = self.analyzer.get_analysis_output(user_input)
        self.analysis_steps = self.analyzer.analysis_steps
        return self.analysis_output

    def _txt_to_sql_analysis(self, user_input: str, analysis_context: str = None):
        self.analyzer = TxttoSQLFactory(
            model=self._analysis_model,
            db_connector=self.database_connector,
            logger=self.logger,
            context=analysis_context,
            model_kwargs=self.analysis_model_kwargs,
            vector_store=self.vector_store,
        )
        self.analysis_output = self.analyzer.get_analysis_output(user_input)
        self.analysis_steps = self.analyzer.analysis_steps
        return self.analysis_output

    def insights(
        self,
        user_input: str,
        insights_context: Optional[str] = None,
        n_insights: Optional[int] = 3,
    ) -> str:
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
                        analysis_context=self.analysis_steps,
                        analysis_output=_format_analysis_output(self.analysis_output),
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
    ) -> str:
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
                analysis_output=f"Analysis output:\n{_format_analysis_output(self.analysis_output)}",
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
    ) -> str:
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
        outputs: list[Literal["insights", "recommendations", "tasks"]] = None,
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
            outputs = ["insights", "recommendations", "tasks"]
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
            "insights": self.insights_output,
            "recommendations": self.recommendations_output,
            "tasks": self.tasks_output,
        }


def _format_analysis_output(output_df, name: str = None) -> str:
    if isinstance(output_df, pd.Series):
        output_df = output_df.to_frame()
    if isinstance(output_df, list):
        return "\n".join([_format_analysis_output(df) for df in output_df])
    if isinstance(output_df, dict):
        return "\n".join(
            [_format_analysis_output(df, name) for name, df in output_df.items()]
        )
    if not isinstance(output_df, pd.DataFrame):
        return str(output_df)

    name = name or "Dataframe"
    if output_df.size > 100:
        df_display = pd.concat([output_df.head(50), output_df.tail(50)], axis=0)
        df_string = f"{name} snapshot:\n{_df_to_string(df_display)}\n\nOutput of `df.describe()`:\n{_df_to_string(output_df.describe())}"
    else:
        df_string = f"{name}:\n{_df_to_string(output_df)}"
    return df_string


def _df_to_string(output_df: pd.DataFrame) -> str:
    # convert all datetime columns to datetime objects
    datetimecols = [
        col
        for col in output_df.columns.tolist()
        if ("date" in col.lower() or "time" in col.lower())
        and output_df[col].dtype != np.number
    ]
    if "timestamp" in output_df.columns and "timestamp" not in datetimecols:
        datetimecols.append("timestamp")
    for col in datetimecols:
        output_df[col] = output_df[col].astype(dtype="datetime64[ns]", errors="ignore")
        output_df.loc[:, col] = pd.to_datetime(output_df[col], errors="ignore")

    datetimecols = output_df.select_dtypes(include=["datetime64"]).columns.tolist()
    formatters = {col: _format_date for col in datetimecols}
    return output_df.to_string(
        float_format="{:,.2f}".format,
        formatters=formatters,
        na_rep="None",
    )


def _format_date(date: pd.Timestamp):
    return date.strftime("%d %b %Y %H:%M")


def _format_df_dict_head(df_dict: dict[pd.DataFrame]) -> str:
    return "\n".join([f"{name}:\n{df.head()}\n" for name, df in df_dict.items()])
