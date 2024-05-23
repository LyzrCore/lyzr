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
from lyzr.data_analyzr.utils import iterate_llm_calls
from lyzr.data_analyzr.models import FactoryBaseClass
from lyzr.data_analyzr.db_connector import DatabaseConnector
from lyzr.base.base import ChatMessage, UserMessage, SystemMessage
from lyzr.data_analyzr.vector_store_utils import ChromaDBVectorStore
from lyzr.data_analyzr.analysis_handler.utils import extract_sql, handle_analysis_output


class TxttoSQLFactory(FactoryBaseClass):

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
        **llm_kwargs,  # model_kwargs: dict
    ):
        super().__init__(
            llm=llm,
            logger=logger,
            context=context,
            vector_store=vector_store,
            max_retries=3 if max_retries is None else max_retries,
            time_limit=30 if time_limit is None else time_limit,
            auto_train=auto_train,
            llm_kwargs=llm_kwargs,
        )
        self.connector = db_connector

    def run_analysis(
        self,
        user_input: str,
        **kwargs,  # max_tries=3, time_limit=30, auto_train=True, for_plotting=False
    ) -> Union[str, pd.DataFrame, dict[str, pd.DataFrame], None]:
        user_input = user_input.strip()
        if user_input is None or user_input == "":
            raise MissingValueError("A user input is required for analysis.")
        for_plotting = kwargs.pop("for_plotting", False)
        messages = self.get_prompt_messages(user_input, for_plotting)
        self.extract_and_run_sql = iterate_llm_calls(
            max_retries=kwargs.pop("max_retries", self.params.max_retries),
            llm=self.llm,
            llm_messages=messages,
            logger=self.logger,
            log_messages={
                "start": f"Starting SQL analysis for query: {user_input}",
                "end": f"Ending SQL analysis for query: {user_input}",
            },
            time_limit=kwargs.pop("time_limit", self.params.time_limit),
        )(self.extract_and_run_sql)
        output = self.extract_and_run_sql()
        if output is None:
            analysis_output = None
            self.code = None
        else:
            analysis_output, self.code = output
        self.output = handle_analysis_output(analysis_output)
        self.guide = self.code
        # Auto-training
        if (
            kwargs.pop("auto_train", self.params.auto_train)
            and self.output is not None
            and len(self.output) > 0
        ):
            self.add_training_data(user_input, self.code)
        return self.output

    def get_prompt_messages(
        self, user_input: str, for_plotting: bool = False
    ) -> list[ChatMessage]:
        question_sql_list = self.vector_store.get_similar_question_sql(user_input)
        ddl_list = self.vector_store.get_related_ddl(user_input)
        doc_list = self.vector_store.get_related_documentation(user_input)
        messages = self._get_sql_prompt(
            user_input=user_input,
            question_sql_list=question_sql_list,
            ddl_list=ddl_list,
            doc_list=doc_list,
            for_plotting=for_plotting,
        )
        # llm_response = self.llm.run(messages, **kwargs).message.content
        return messages

    def _get_sql_prompt(
        self,
        user_input: str,
        question_sql_list: list,
        ddl_list: list,
        doc_list: list,
        for_plotting: bool = False,
    ) -> list[ChatMessage]:
        system_message_sections = ["context", "external_context"]
        system_message_sections.append("task" if not for_plotting else "plotting_task")
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

    def extract_and_run_sql(self, llm_response):
        sql_query, analysis_output = None, None
        sql_query = extract_sql(llm_response)
        self.logger.info(f"Extracted SQL query:\n{sql_query}")
        if "CREATE TABLE" in sql_query:
            analysis_output = self._handle_create_table_sql(sql_query)
        else:
            analysis_output = self.connector.run_sql(sql_query)
        return (analysis_output, sql_query)

    def _handle_create_table_sql(self, sql_query: str):
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

    def add_training_data(self, user_input: str, sql_query: str):
        if user_input is None or user_input.strip() == "":
            user_input = self._generate_question(sql_query)
        if sql_query is not None and sql_query.strip() != "":
            self.logger.info("Saving data for next training round\n")
            self.vector_store.add_training_plan(question=user_input, sql=sql_query)

    def _generate_question(self, sql: str, **kwargs) -> str:
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
