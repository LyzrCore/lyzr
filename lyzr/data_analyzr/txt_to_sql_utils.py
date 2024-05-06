# standard library imports
import logging

# local imports
from lyzr.base.llm import LiteLLM
from lyzr.base.prompt import LyzrPromptFactory
from lyzr.base.errors import MissingValueError
from lyzr.data_analyzr.utils import iterate_llm_calls
from lyzr.data_analyzr.output_handler import extract_sql
from lyzr.data_analyzr.db_connector import DatabaseConnector
from lyzr.base.base import ChatMessage, UserMessage, SystemMessage
from lyzr.data_analyzr.vector_store_utils import ChromaDBVectorStore


class TxttoSQLFactory:

    def __init__(
        self,
        llm: LiteLLM,
        db_connector: DatabaseConnector,
        logger: logging.Logger,
        context: str,
        vector_store: ChromaDBVectorStore,
        max_tries: int = None,
        time_limit: int = None,
        auto_train: bool = None,
        **llm_kwargs,  # model_kwargs: dict
    ):
        self.llm = llm
        model_kwargs = dict(seed=123, temperature=0.1, top_p=0.5)
        model_kwargs.update(llm_kwargs)
        self.llm.set_model_kwargs(model_kwargs=model_kwargs)
        self.context = context.strip() + "\n\n" if context.strip() != "" else ""
        self.logger = logger
        self.connector = db_connector
        self.vector_store = vector_store
        if self.vector_store is None:
            raise MissingValueError("vector_store")
        self.analysis_output, self.sql_query = None, None
        self.max_tries = max_tries if max_tries is not None else 3
        self.time_limit = time_limit if time_limit is not None else 30
        self.auto_train = auto_train if auto_train is not None else True

    def run_analysis(
        self,
        user_input: str,
        **kwargs,  # max_tries=3, time_limit=30, auto_train=True, for_plotting=False
    ):
        user_input = user_input.strip()
        if user_input is None or user_input == "":
            raise MissingValueError("A user input is required for analysis.")
        for_plotting = kwargs.pop("for_plotting", False)
        messages = self._generate_sql_messages(user_input, for_plotting)
        self.extract_and_run_sql = iterate_llm_calls(
            max_tries=kwargs.pop("max_tries", self.max_tries),
            llm=self.llm,
            llm_messages=messages,
            logger=self.logger,
            log_messages={
                "start": f"Starting SQL analysis for query: {user_input}",
                "end": f"Ending SQL analysis for query: {user_input}",
            },
            time_limit=kwargs.pop("time_limit", self.time_limit),
        )(self.extract_and_run_sql)
        self.analysis_output, self.sql_query = self.extract_and_run_sql()
        # Auto-training
        if (
            kwargs.pop("auto_train", True)
            and self.analysis_output is not None
            and len(self.analysis_output) > 0
        ):
            self.add_training_data(user_input, self.sql_query)
        return self.analysis_output

    def extract_and_run_sql(self, llm_response):
        sql_query = extract_sql(llm_response)
        analysis_output = self.connector.run_sql(sql_query)
        return (analysis_output, sql_query)

    def _generate_sql_messages(
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

        system_message_sections.append("closing")
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

    def add_training_data(self, sql_query: str, user_input: str):
        if user_input is None or user_input.strip() == "":
            user_input = self._generate_question(self.sql_query)
        if sql_query is not None and sql_query.strip() != "":
            self.logger.info("Saving data for next training round\n")
            self.vector_store.add_training_plan(question=user_input, sql=self.sql_query)
