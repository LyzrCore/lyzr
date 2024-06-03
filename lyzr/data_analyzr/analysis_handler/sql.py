"""
TxttoSQLFactory class - generate and execute SQL queries from text inputs.
"""

# standard library imports
import re
import logging
import traceback
from typing import Union

# third-party imports
import pandas as pd

# local imports
from lyzr.base.llm import LiteLLM
from lyzr.base.prompt import LyzrPromptFactory
from lyzr.base.errors import MissingValueError
from lyzr.data_analyzr.models import FactoryBaseClass
from lyzr.data_analyzr.db_connector import DatabaseConnector
from lyzr.base.base import ChatMessage, UserMessage, SystemMessage
from lyzr.data_analyzr.vector_store_utils import ChromaDBVectorStore
from lyzr.data_analyzr.analysis_handler.utils import (
    extract_sql,
    iterate_llm_calls,
    handle_analysis_output,
)


class TxttoSQLFactory(FactoryBaseClass):
    """
    TxttoSQLFactory is a class that facilitates the conversion of natural language text into SQL queries using a language model.

    This class integrates with a database connector to execute the generated SQL queries and provides logging and context
    management functionalities. It also supports automatic training data generation and management through a vector store.

    Attributes:
        connector (DatabaseConnector): The database connector for executing generated SQL queries.

    Methods:
        generate_output(user_input: str, **kwargs) -> Union[str, pd.DataFrame, dict[str, pd.DataFrame], None]:
            Run an analysis on the provided user input by generating and executing SQL queries.

        get_prompt_messages(user_input: str) -> list[ChatMessage]:
            Generates a list of chat messages based on the user's input.

        _get_sql_prompt(user_input: str, question_sql_list: list, ddl_list: list, doc_list: list) -> list[ChatMessage]:
            Generates a list of ChatMessages based on the provided user input and context.

        extract_and_execute_code(llm_response: str):
            Extracts an SQL query from the given LLM response and executes it.

        _handle_create_table_sql(sql_query: str):
            Handles the execution of a SQL query when table creation is involved.

        auto_train(user_input: str, code: str, **kwargs):
            Adds the user input and generated SQL to the vector store if the auto_train flag is set.

        _generate_question(sql: str, **kwargs) -> str:
            Generates a question based on the provided SQL query.
    """

    def __init__(
        self,
        llm: LiteLLM,
        db_connector: DatabaseConnector,
        logger: logging.Logger,
        context: str,
        vector_store: ChromaDBVectorStore,
        max_retries: int = None,
        time_limit: int = None,
        auto_train: bool = None,
        **llm_kwargs,
    ):
        """
        Initialize a TxttoSQLFactory instance.

        Args:
            llm (LiteLLM): The llm instance to be used.
            db_connector (DatabaseConnector): The database connector for executing generated SQL queries.
            logger (logging.Logger): Logger for logging events and errors.
            context (str): The context for the given query. If empty, pass "".
            vector_store (ChromaDBVectorStore): The vector store for managing related queries and database documentation.
            max_retries (int, optional): Maximum number of retries for analysis operations. Defaults to 10 if None.
            time_limit (int, optional): Time limit for analysis in seconds. Defaults to 45 if None.
            auto_train (bool, optional): Whether to automatically add to training data. Defaults to True.
            **llm_kwargs: Additional keyword arguments for the language model.

        Example Usage:
            from lyzr.base.llm import LiteLLM
            from lyzr.data_analyzr.vector_store_utils import ChromaDBVectorStore
            from lyzr.data_analyzr.analysis_handler import TxttoSQLFactory

            llm = LiteLLM.from_defaults(model="gpt-4o")
            logger = logging.getLogger(__name__)
            context = "Context for the analysis."
            db_connector = DatabaseConnector(...)
            vector_store = ChromaDBVectorStore(path="path/to/vector_store", logger=logger)
            analysis_factory = TxttoSQLFactory(
                llm=llm,
                db_connector=db_connector,
                logger=logger,
                context=context,
                vector_store=vector_store
            )
        """
        super().__init__(
            llm=llm,
            logger=logger,
            context=context,
            vector_store=vector_store,
            max_retries=10 if max_retries is None else max_retries,
            time_limit=45 if time_limit is None else time_limit,
            auto_train=auto_train,
            llm_kwargs=llm_kwargs,
        )
        self.connector = db_connector

    def generate_output(
        self,
        user_input: str,
        **kwargs,  # max_tries=3, time_limit=30, auto_train=True
    ) -> Union[str, pd.DataFrame, dict[str, pd.DataFrame], None]:
        """
        Run an analysis on the provided user input by generating and executing SQL queries.

        This is the primary method for generating and executing SQL analysis.
            - Processes the user input to generate SQL analysis output.
            - Handles retries and time limits for the SQL extraction and execution process.
            - Uses iterate_llm_calls to handle retries and time limits.

        Args:
            user_input (str): The user's question input for generating SQL queries. Should be in natural language.

        Keyword Args:
            max_retries (int, optional): Maximum number of retries for generating SQL queries. Defaults to self.params.max_retries.
            time_limit (int, optional): Time limit for the analysis process in seconds. Defaults to self.params.time_limit.
            auto_train (bool, optional): Whether to automatically add the analysis to the training data. Defaults to self.params.auto_train.

        Raises:
            MissingValueError: If the user input is empty or None.

        Returns:
            Union[str, pd.DataFrame, dict[str, pd.DataFrame], None]: The result of the SQL analysis.

        Example usage:
            output = analysis_factory.generate_output("Your question here.")
            print(output)
        """
        user_input = user_input.strip()
        if user_input is None or user_input == "":
            raise MissingValueError("A user input is required for analysis.")
        messages = self.get_prompt_messages(user_input)
        self.extract_and_execute_code = iterate_llm_calls(
            max_retries=kwargs.pop("max_retries", self.params.max_retries),
            llm=self.llm,
            llm_messages=messages,
            logger=self.logger,
            log_messages={
                "start": f"Starting SQL analysis for query: {user_input}",
                "end": f"Ending SQL analysis for query: {user_input}",
            },
            time_limit=kwargs.pop("time_limit", self.params.time_limit),
        )(self.extract_and_execute_code)
        self.output = handle_analysis_output(self.extract_and_execute_code())
        self.auto_train(user_input, self.code, **kwargs)
        return self.output

    def get_prompt_messages(self, user_input: str) -> list[ChatMessage]:
        """
        Generates a list of chat messages based on the user's input.

        Retrieve similar SQL questions, related DDL statements, and relevant documentation
        from the vector store based on the user's input. The list of messages includes system
        messages with context, DDL statements, documentation, and historical examples, followed
        by the user's input.

        Args:
            user_input (str): The input provided by the user.

        Returns:
            list[ChatMessage]: A list of chat messages generated based on the user's input.
        """
        question_sql_list = self.vector_store.get_related_sql_queries(user_input)
        ddl_list = self.vector_store.get_related_ddl(user_input)
        doc_list = self.vector_store.get_related_documentation(user_input)
        messages = self._get_sql_prompt(
            user_input=user_input,
            question_sql_list=question_sql_list,
            ddl_list=ddl_list,
            doc_list=doc_list,
        )
        return messages

    def _get_sql_prompt(
        self,
        user_input: str,
        question_sql_list: list,
        ddl_list: list,
        doc_list: list,
    ) -> list[ChatMessage]:
        """
        Generates a list of ChatMessages based on the provided user input and context.

        Args:
            user_input (str): The user's input or query that needs to be converted to SQL.
            question_sql_list (list): A list of dictionaries containing example questions and their corresponding SQL queries.
            ddl_list (list): A list of Data Definition Language (DDL) statements relevant to the context.
            doc_list (list): A list of additional documentation or context that might help in generating the SQL query.

        Returns:
            list[ChatMessage]: A list of ChatMessage objects that form the complete prompt for generating the SQL query.
        """
        system_message_sections = ["context", "external_context", "task"]
        system_message_dict = {"context": self.context}

        if len(ddl_list) > 0:
            system_message_sections.append("ddl_addition_text")
            system_message_dict["ddl"] = ""
            for ddl_item in ddl_list:
                system_message_dict["ddl"] += f"{ddl_item}\n"

        if len(doc_list) > 0:
            system_message_sections.append("doc_addition_text")
            system_message_dict["doc"] = ""
            for doc_item in doc_list:
                system_message_dict["doc"] += f"{doc_item}\n"
        if len(question_sql_list) > 0:
            system_message_sections.append("history")
        messages = [
            LyzrPromptFactory(
                name="txt_to_sql",
                prompt_type="system",
                use_sections=system_message_sections,
            ).get_message(**system_message_dict)
        ]

        for example in question_sql_list:
            if example is not None:
                if "question" in example and "sql" in example:
                    messages.append(UserMessage(content=example["question"]))
                    messages.append(SystemMessage(content=example["sql"]))

        messages.append(UserMessage(content=user_input))
        return messages

    def extract_and_execute_code(self, llm_response: str):
        """
        Extracts an SQL query from the given LLM response and executes it.
        To be used as a callback function for iterate_llm_calls.

        Args:
            llm_response (str): The response from the language model containing the SQL query.

        Returns:
            Any: The result of the executed SQL query.

        Logs:
            - Info: Extracted SQL query.

        Procedure:
            - Extracts the SQL query from the LLM response.
            - Checks if the SQL query is a CREATE TABLE statement.
            - Executes the SQL query using the database connector.
            - If the SQL query is a CREATE TABLE statement, passes control to _handle_create_table_sql.
        """
        sql_query, analysis_output = None, None
        sql_query = extract_sql(llm_response)
        self.logger.info(f"Extracted SQL query:\n{sql_query}")
        match = re.search("CREATE TABLE", sql_query, re.IGNORECASE)
        if match is not None:
            analysis_output = self._handle_create_table_sql(sql_query)
        else:
            analysis_output = self.connector.run_sql(sql_query)
        self.code = sql_query
        self.guide = sql_query
        return analysis_output

    def _handle_create_table_sql(self, sql_query: str):
        """
        Handles the execution of a SQL query when table creation is involved.

        Tries to execute the SQL query without the CREATE TABLE part first.
        If that fails, it retries the original SQL query.

        Args:
            sql_query (str): The SQL query to be executed.

        Returns:
            Any: The result of the executed SQL query.

        Raises:
            Logs an error and attempts to re-run the original SQL query if an exception occurs during execution.

        Logs:
            - Error: When an exception occurs during SQL execution.
        """
        match = re.search("SELECT", sql_query, re.IGNORECASE)
        if match is not None:
            sub_query = sql_query[match.start() :]
        else:
            sub_query = sql_query
        try:
            return self.connector.run_sql(sub_query)
        except Exception as e:
            self.logger.error(
                f"Error running SQL sub-query.{e.__class__.__name__}: {e}",
                extra={
                    "input_kwargs": {
                        "sql_query": sql_query,
                        "sub_query": sub_query,
                    },
                    "traceback": traceback.format_exc().splitlines(),
                },
            )
            return self.connector.run_sql(sql_query)

    def auto_train(self, user_input: str, code: str, **kwargs):
        """
        Adds the user input and generated SQL to the vector store if the auto_train flag is set.

        Args:
            user_input (str): The user's input query.
            code (str): The SQL query generated from the user input.
            auto_train (bool): Whether to automatically add the analysis to the training data,
                defaults to self.params.auto_train.

        Returns:
            Any: The output generated by the model, if available.

        Procedure:
            - Checks if the auto_train flag is set and the output is not None.
            - Generates a question based on the SQL query if the user input is empty or None.
            - Adds the user input and SQL query to the training data in the vector store.
        """
        if (
            kwargs.pop("auto_train", self.params.auto_train)
            and self.output is not None
            and len(self.output) > 0
        ):
            if user_input is None or user_input.strip() == "":
                user_input = self._generate_question(code)
            if code is not None and code.strip() != "":
                self.vector_store.add_training_data(question=user_input, sql=code)
        return self.output

    def _generate_question(self, sql: str, **kwargs) -> str:
        """
        Generates a question based on the provided SQL query.
        To be used when the user input is empty or None.

        Args:
            sql (str): The SQL query for which to generate a question.
            **kwargs: Additional keyword arguments to pass to the llm.

        Returns:
            str: The generated question based on the SQL query.
        """
        kwargs["max_tokens"] = 500
        output = self.llm.run(
            messages=[
                LyzrPromptFactory(
                    name="sql_question_gen", prompt_type="system"
                ).get_message(),
                UserMessage(content=sql),
            ],
            **kwargs,
        )
        return output.message.content
