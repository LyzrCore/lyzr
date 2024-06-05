"""
DataAnalyzr class, for conversational data analysis.
"""

# standard library imports
import json
import time
from typing import Literal, Optional, Union

# third-party imports
import pandas as pd

# local imports
from lyzr.data_analyzr.models import (
    AnalysisTypes,
    OutputTypes,
    ParamsDict,
    ContextDict,
    VectorStoreConfig,
)
from lyzr.base.prompt import LyzrPromptFactory
from lyzr.base.llm import LyzrLLMFactory, LiteLLM
from lyzr.data_analyzr.db_models import SupportedDBs, DataConfig


class DataAnalyzr:
    """
    DataAnalyzr is a class designed to perform data analysis based on natural language input.
    It leverages large language models (LLMs) for generating insights, visualizations, recommendations, and tasks based on user input.

    Attributes:
        analysis_type (Literal["sql", "ml", "skip"]): The type of analysis to be performed.
        api_key (str): API key for accessing LLM services.
        params (ParamsDict): Dictionary of class parameters.
        generator_llm (LiteLLM): LLM instance for generating analysis.
        analysis_llm (LiteLLM): LLM instance for performing analysis.
        context (ContextDict): Context for analysis and response generation.
        logger (logging.Logger): Logger instance for logging messages.
        analysis_guide (str): The guide for the analysis process.
        analysis_code (str): The code generated for the analysis.
        analysis_output (Union[str, pd.DataFrame, dict[str, pd.DataFrame], None]): The output of the analysis process.
        plot_code (str): The code generated for the visualization.
        plot_output (str): The output of the visualization process.
        insights_output (str): The generated insights as a string.
        recommendations_output (str): The generated recommendations in the specified format.
        tasks_output (str): The generated tasks as a string.
        ai_queries_output (dict[str, list]): The generated AI queries for different types of analysis.

    Methods:
        get_data(db_type, db_config, vector_store_config):
            Retrieve data from the specified database type using the provided configuration.
        analysis(user_input, analysis_context, time_limit, max_retries, auto_train):
            Perform an analysis based on the provided user input and analysis parameters.
        visualisation(user_input, plot_context, time_limit, max_retries, auto_train, plot_path):
            Generate a visualization based on the provided user input and context.
        insights(user_input, insights_context, n_insights):
            Generate insights based on the provided user input and optional context.
        recommendations(user_input, from_insights, recs_format, recommendations_context, n_recommendations, output_type):
            Generate recommendations based on the provided user input and analysis insights.
        tasks(user_input, tasks_context, n_tasks):
            Generate a list of tasks based on the provided user input and context.
        ask(user_input, outputs, plot_path, recommendations_params, counts, context, **kwargs):
            Ask a question and generate various outputs based on the provided user input and optional parameters.
        ai_queries(context):
            Generate AI-based queries for data analysis based on the provided context.
    """

    def __init__(
        self,
        analysis_type: Literal["sql", "ml", "skip"],
        api_key: Optional[str] = None,
        class_params: Optional[dict] = None,
        generator_llm: Optional[LiteLLM] = None,
        analysis_llm: Optional[LiteLLM] = None,
        context: Optional[Union[str, dict]] = None,
        log_params: Optional[dict] = None,
    ):
        """
        Initialize a DataAnalyzr instance.

        Args:
            analysis_type (Literal["sql", "ml", "skip"]): The type of analysis to be performed.

        Keyword Args:
            api_key (Optional[str], optional): API key for accessing LLM services. May also be set as an environment variable.
            class_params (Optional[dict], optional): Dictionary of class parameters.
                - max_retries (int): Maximum number of retries for LLM calls. Defaults to None.
                - time_limit (int): Time limit for LLM calls. Defaults to None.
                - auto_train (bool): Whether to train the LLM model. Defaults to True.
            generator_llm (Optional[LiteLLM], optional): LLM instance for generating analysis. Defaults to OpenAI's gpt-4o.
            analysis_llm (Optional[LiteLLM], optional): LLM instance for performing analysis. Defaults to OpenAI's gpt-4o.
            context (Optional[Union[str, dict]], optional): Context for analysis and response generation. Defaults to "".
            log_params (Optional[dict], optional): Dictionary of logging parameters.
                - log_filename (str): Name of the log file. Defaults to "dataanalyzr.csv".
                - log_level (str): Level of logging. Defaults to "INFO".
                - print_log (bool): Whether to print logs to console. Defaults to False.

        Raises:
            ValueError: If the provided `analysis_type` is not one of "sql", "ml", or "skip".

        Example:
            from lyzr.base.llm import LiteLLM
            from lyzr.data_analyzr.data_analyzr import DataAnalyzr

            analysis_type = "sql"
            log_params = {"log_filename": "dataanalyzr.log", "log_level": "DEBUG", "print_log": True}

            data_analyzr = DataAnalyzr(
                analysis_type=analysis_type,
                api_key=api_key,
                class_params=class_params,
                generator_llm=generator_llm,
                analysis_llm=analysis_llm,
                context=context,
                log_params=log_params,
            )
        """
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

        self.context = ContextDict().validate(context)
        (
            self.df_dict,
            self.database_connector,
            self.vector_store,
            self.analysis_code,
            self.analysis_guide,
            self.analysis_output,
            self.plot_code,
            self.plot_output,
            self.insights_output,
            self.recommendations_output,
            self.tasks_output,
            self.ai_queries_output,
        ) = (None,) * 12

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
        db_config: dict,
        vector_store_config: dict = {},
    ):
        """
        Retrieve data from the specified database type using the provided configuration.

        Args:
            db_type (Literal["files", "redshift", "postgres", "sqlite"]): The type of database to connect to.
            db_config (dict): Configuration dictionary for the database connection.
            vector_store_config (dict, optional): Configuration dictionary for the vector store. Defaults to an empty dictionary.

        Raises:
            ValueError: If data_config is not a dictionary.

        Example:
            db_config = {
                "host": "localhost",
                "port": 5432,
                "user": "username",
                "password": "password",
                "database": "dbname"
            }
            vector_store_config = {
                "path": "path/to/vector_store"
            }
            data_analyzr.get_data(
                db_type="postgres",
                db_config=db_config,
                vector_store_config=vector_store_config
            )
        """
        from pydantic import TypeAdapter
        from lyzr.data_analyzr.file_utils import get_db_details

        if not isinstance(db_config, dict):
            raise ValueError("data_config must be a dictionary.")
        db_config["db_type"] = SupportedDBs(db_type.lower().strip())
        self.database_connector, self.df_dict, self.vector_store = get_db_details(
            analysis_type=self.analysis_type,
            db_type=db_config["db_type"],
            db_config=TypeAdapter(DataConfig).validate_python(db_config),
            vector_store_config=VectorStoreConfig(**vector_store_config),
            logger=self.logger,
        )

    def analysis(
        self,
        user_input: str,
        analysis_context: str = None,
        time_limit: int = None,
        max_retries: int = None,
        auto_train: bool = None,
    ) -> Union[str, pd.DataFrame, dict[str, pd.DataFrame], None]:
        """
        Perform an analysis based on the provided user input and analysis parameters.

        This method determines the type of analysis to be performed (SQL, or Pythonic) and executes it.
        If the analysis type is set to skip, it fetches dataframes from the database and returns them
        without performing any analysis.

        Args:
            user_input (str): The input string provided by the user for analysis.
            analysis_context (str): The context for the analysis. Defaults to "".
            time_limit (int): Time limit for the analysis execution in seconds. Defaults to 45.
            max_retries (int): Maximum number of retries for the analysis. Defaults to 10.
            auto_train (bool): Whether to automatically add the analysis to training data. Defaults to True.

        Returns:
            Union[str, pd.DataFrame, dict[str, pd.DataFrame], None]: The output of the analysis.

        Example:
            output = data_analyzr.analysis(
                user_input="Analyze the sales data.",
                analysis_context="Sales data analysis for Q1 2023",
                time_limit=60,
                max_retries=5,
                auto_train=True
            )
            print(output)
        """
        if self.analysis_type is AnalysisTypes.skip:
            self.logger.info("No analysis performed.")
            self.analysis_guide = "No analysis performed."
            self.analysis_output = None
            self.analysis_code = None
            return self.analysis_output
        if analysis_context is None:
            analysis_context = ""
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
            from lyzr.data_analyzr.analysis_handler import TxttoSQLFactory

            analyser = TxttoSQLFactory(
                **analyser_args,
                db_connector=self.database_connector,
            )
        if self.analysis_type is AnalysisTypes.ml:
            from lyzr.data_analyzr.analysis_handler import PythonicAnalysisFactory

            analyser = PythonicAnalysisFactory(
                **analyser_args,
                df_dict=self.df_dict,
            )
        self.analysis_output = analyser.generate_output(user_input)
        self.analysis_guide = analyser.guide
        self.analysis_code = analyser.code
        return self.analysis_output

    def visualisation(
        self,
        user_input: str,
        plot_context: str = None,
        time_limit: int = None,
        max_retries: int = None,
        auto_train: bool = None,
        plot_path: str = None,
    ) -> str:
        """
        Generate a visualisation based on the provided user input and context.

        This method uses the PlotFactory to create visualizations from the analysis output or provided dataframes.
        It handles retries and time limits for the visualization process and saves the plot to a specified path.

        Args:
            user_input (str): The input question.
            plot_context (str): The context for the plot. Defaults to "".
            time_limit (int): Time limit for the visualization process in seconds. Defaults to 60.
            max_retries (int): Maximum number of retries for the visualization process. Defaults to 10.
            auto_train (bool): Whether to automatically add the visualization to training data. Defaults to True.
            plot_path (str, optional): Path to save the generated plot. Defaults to generated_plots/<random-string>.png.

        Returns:
            The output of the visualization process, which could be a plot object or a path to the saved plot.

        Example:
            saved_plot_path = data_analyzr.visualisation(
                user_input="Plot the distribution of column A.",
                plot_context="Distribution plot",
                time_limit=60,
                max_retries=5,
                auto_train=True,
                plot_path="path/to/save/plot.png"
            )
            from PIL import Image
            Image.open(saved_plot_path).show()
        """
        from lyzr.data_analyzr.analysis_handler import PlotFactory

        if plot_context is None:
            plot_context = ""
        if self.df_dict is None:
            data_kwargs = {"connector": self.database_connector}
        else:
            data_kwargs = {"df_dict": self.df_dict}
        if isinstance(self.analysis_output, pd.DataFrame):
            data_kwargs["analysis_output"] = {"analysis_output": self.analysis_output}
        elif isinstance(self.analysis_output, dict):
            data_kwargs["analysis_output"] = self.analysis_output
        self.plot_output = None
        plotter = PlotFactory(
            llm=self.analysis_llm,
            logger=self.logger,
            context=plot_context,
            data_kwargs=data_kwargs,
            vector_store=self.vector_store,
            time_limit=time_limit,
            max_retries=max_retries,
            auto_train=auto_train,
        )
        self.plot_output = plotter.generate_output(
            user_input=user_input, plot_path=plot_path
        )
        self.plot_code = plotter.code
        return self.plot_output

    def insights(
        self,
        user_input: str,
        insights_context: Optional[str] = None,
        n_insights: Optional[int] = 3,
    ) -> str:
        """
        Generate insights based on the provided user input and optional context.

        Args:
            user_input (str): The input string provided by the user for generating insights.
            insights_context (Optional[str], optional): Additional context for generating insights. Defaults to "".
            n_insights (Optional[int], optional): Number of insights to generate. Defaults to 3.

        Returns:
            str: The generated insights as a string.

        Example:
            insights = data_analyzr.insights("What are the key trends?")
            print(insights)
        """
        from lyzr.data_analyzr.utils import format_analysis_output

        if insights_context is None:
            insights_context = ""
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
                    analysis_output=(
                        format_analysis_output(output_df=self.analysis_output)
                        if self.analysis_output is not None
                        else self.vector_store.get_related_documentation(user_input)
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
        user_input: str,
        from_insights: Optional[bool] = True,
        recs_format: Optional[dict] = None,
        recommendations_context: Optional[str] = None,
        n_recommendations: Optional[int] = 3,
        output_type: Optional[Literal["text", "json"]] = "text",
    ) -> str:
        """
        Generate recommendations based on the provided user input and analysis insights.

        This method generates actionable recommendations by leveraging the analysis output and optional insights.
        It can format the recommendations in either text or JSON format.

        Args:
            user_input (Optional[str]): The input string provided by the user for generating recommendations.
            from_insights (Optional[bool]): Whether to include insights from the analysis in the recommendations. Defaults to True.
            recs_format (Optional[dict]): The format for the recommendations if output_type is "json". Defaults to None.
            recommendations_context (Optional[str]): Additional context to be included in the recommendations. Defaults to None.
            n_recommendations (Optional[int]): The number of recommendations to generate. Defaults to 3.
            output_type (Optional[Literal["text", "json"]]): The format of the output recommendations.
                Can be "text" or "json". Defaults to "text".

        Returns:
            str: The generated recommendations in the specified format.

        Example:
            recommendations = data_analyzr.recommendations(user_input="How can we improve our sales?")
            print(recommendations)
        """
        from lyzr.data_analyzr.utils import format_analysis_output

        system_message_sections = ["context"]
        system_message_dict = {}
        llm_params = dict(
            temperature=0.3,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            logger=self.logger,
        )
        df_details = (
            format_analysis_output(output_df=self.analysis_output)
            if self.analysis_output is not None
            else self.vector_store.get_related_documentation(user_input)
        )
        user_message_dict = {
            "user_input": user_input,
            "analysis_output": f"Analysis output:\n{df_details}",
        }
        if recommendations_context is None:
            recommendations_context = ""
        system_message_dict["context"] = recommendations_context

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
        user_input: str,
        tasks_context: Optional[str] = None,
        n_tasks: Optional[int] = 5,
    ) -> str:
        """
        Generate a list of tasks based on the provided user input and context.

        Args:
            user_input (Optional[str]): The input string provided by the user for generating tasks. Defaults to None.
            tasks_context (Optional[str]): The context for the tasks to be generated. Defaults to "".
            n_tasks (Optional[int]): The number of tasks to generate. Defaults to 5.

        Returns:
            str: The generated tasks as a string.

        Example:
            tasks_output = data_analyzr.tasks("Analyze sales data")
            print(tasks_output)
        """
        if tasks_context is None:
            tasks_context = ""
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
    ) -> dict[str, str]:
        """
        Ask a question and generate various outputs based on the provided user input and optional parameters.
        This is the primary method for generating responses from the DataAnalyzr instance.

        Args:
            user_input (str): The input string provided by the user for generating outputs.
            outputs (list, optional): List of output types to generate. Defaults to all output types if not provided.
                - "visualisation"
                - "insights"
                - "recommendations"
                - "tasks"
            plot_path (str, optional): Path to save the generated plot. Defaults to generated_plots/<random-string>.png.
            recommendations_params (dict, optional): Parameters for generating recommendations.
                - from_insights (bool): Whether to generate recommendations from insights. Defaults to True.
                - json_format (bool): Whether to format recommendations as JSON. Defaults to None.
                - output_type (str): The format of the recommendations output. Defaults to "text".
            counts (dict, optional): Dictionary specifying the number of insights, recommendations, and tasks to generate.
                - insights (int): Number of insights to generate. Defaults to 3.
                - recommendations (int): Number of recommendations to generate. Defaults to 3.
                - tasks (int): Number of tasks to generate. Defaults to 5.
            context (dict, optional): Dictionary providing context for analysis, insights, recommendations, and tasks. Defaults to self.context.
                - analysis (str): Context for the analysis.
                - visualisation (str): Context for the visualisation.
                - insights (str): Context for the insights.
                - recommendations (str): Context for the recommendations.
                - tasks (str): Context for the tasks.
            **kwargs: Additional keyword arguments to customize the analysis process.
                - rerun_analysis (bool): Whether to rerun the analysis. Defaults to True.
                - max_retries (int): Maximum number of retries for analysis. Defaults to self.params.max_retries.
                - time_limit (int): Time limit for the analysis in seconds. Defaults to self.params.time_limit.
                - auto_train (bool): Whether to automatically add to training data. Defaults to self.params.auto_train.

        Returns:
            dict[str, str]: A dictionary containing the generated outputs.
                - "visualisation": Path to the generated visualization image.
                - "insights": The generated insights output.
                - "recommendations": The generated recommendations output.
                - "tasks": The generated tasks output.

        Raises:
            AssertionError: If user_input is not provided or is an empty string.

        Example:
            outputs = data_analyzr.ask(
                user_input="What are the key trends in the sales data?",
                outputs=["visualisation", "insights", "recommendations"],
                plot_path="path/to/save/plot.png",
                recommendations_params={"from_insights": True, "json_format": True},
                counts={"insights": 5, "recommendations": 3, "tasks": 2},
                context={"analysis": "sales analysis context"},
                rerun_analysis=True,
                max_retries=5,
                time_limit=60,
                auto_train=True
            )
            print(outputs)
        """
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
        self.context = self.context.validate(context)
        counts = counts or {}
        recommendations_params = recommendations_params or {}

        time_limit = kwargs.get("time_limit", self.params.time_limit)
        max_retries = kwargs.get("max_retries", self.params.max_retries)
        auto_train = kwargs.get("auto_train", self.params.auto_train)

        if self.analysis_output is None or rerun_analysis:
            self.analysis_output = self.analysis(
                user_input=user_input,
                analysis_context=self.context.analysis,
                time_limit=time_limit,
                max_retries=max_retries,
                auto_train=auto_train,
            )

        if OutputTypes.visualisation in outputs:
            self.plot_output = self.visualisation(
                user_input=user_input,
                plot_context=self.context.visualisation,
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
                insights_context=self.context.insights,
                n_insights=counts.get("insights", 3),
            )
        else:
            self.insights_output = ""

        if OutputTypes.recommendations in outputs or OutputTypes.tasks in outputs:
            self.recommendations_output = self.recommendations(
                user_input=user_input,
                from_insights=recommendations_params.get("from_insights", True),
                recs_format=recommendations_params.get("json_format", None),
                recommendations_context=self.context.recommendations,
                n_recommendations=counts.get("recommendations", 3),
                output_type=recommendations_params.get("output_type", "text"),
            )
        else:
            self.recommendations_output = ""

        if OutputTypes.tasks in outputs:
            self.tasks_output = self.tasks(
                user_input=user_input,
                tasks_context=self.context.tasks,
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
    ) -> dict[str, list[str]]:
        """
        Generate AI-based queries for data analysis based on the provided context.

        Args:
            context (Optional[str]): The context for generating queries. Defaults to "".

        Returns:
            dict[str, list[str]]: A dictionary containing the generated AI queries for different types of analysis.

        Example:
            queries = data_analyzr.ai_queries(context="Analyze sales data trends.")
            print(queries)
        """

        context = self.context.insights if context is None else context
        schema = {
            "type_of_analysis1": ["query1", "query2", "query3", "query4"],
            "type_of_analysis2": ["query1", "query2", "query3", "query4"],
            "type_of_analysis3": ["query1", "query2", "query3", "query4"],
        }
        messages = [
            LyzrPromptFactory(name="ai_queries", prompt_type="system").get_message(
                context=context,
                schema=json.dumps(schema, indent=4),
            ),
            LyzrPromptFactory(name="ai_queries", prompt_type="user").get_message(
                df_details=self.vector_store.get_related_documentation(context)
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
