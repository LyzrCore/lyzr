# standard library imports
import time
import logging
import traceback

# local imports
from lyzr.base.prompt import LyzrPromptFactory
from lyzr.base.errors import MissingValueError
from lyzr.base.llm import LiteLLM
from lyzr.base.base import ChatMessage, UserMessage, SystemMessage
from lyzr.data_analyzr.output_handler import extract_sql
from lyzr.data_analyzr.db_connector import DatabaseConnector
from lyzr.data_analyzr.vector_store_utils import ChromaDBVectorStore
from lyzr.data_analyzr.utils import run_n_times


class TxttoSQLFactory:

    def __init__(
        self,
        model: LiteLLM,
        db_connector: DatabaseConnector,
        logger: logging.Logger,
        context: str,
        vector_store: ChromaDBVectorStore,
    ):
        self.model = model
        self.model.set_model_kwargs(
            model_kwargs=dict(seed=123, temperature=0.1, top_p=0.5)
        )
        self.context = context.strip() + "\n\n" if context.strip() != "" else ""
        self.logger = logger
        self.connector = db_connector
        self.vector_store = vector_store
        if self.vector_store is None:
            raise MissingValueError("vector_store")
        self.analysis_output, self.analysis_guide = None, None

    def run_complete_analysis(
        self,
        user_input: str,
        auto_train: bool = True,
        **kwargs,
    ):
        start_time = time.time()
        question_sql_list = self.vector_store.get_similar_question_sql(
            user_input, **kwargs
        )
        ddl_list = self.vector_store.get_related_ddl(user_input, **kwargs)
        doc_list = self.vector_store.get_related_documentation(user_input, **kwargs)
        messages = self._get_sql_prompt(
            user_input=user_input,
            question_sql_list=question_sql_list,
            ddl_list=ddl_list,
            doc_list=doc_list,
        )
        # print(messages)
        for _ in range(3):
            try:
                # SQL Generation
                self.analysis_guide = extract_sql(
                    self.model.run(messages=messages, **kwargs).message.content,
                    self.logger,
                )
                self.logger.info(f"Analysis Guide:\n{self.analysis_guide}\n")
                # SQL Execution
                self.analysis_output = self.connector.run_sql(self.analysis_guide)
                self.logger.info(f"Analysis Output:\n{self.analysis_output}\n")
                break
            except RecursionError:
                raise RecursionError(
                    "The request could not be completed. Please wait a while and try again."
                )
            except Exception as e:
                if time.time() - start_time > 30:
                    raise TimeoutError(
                        "The request could not be completed. Please wait a while and try again."
                    )
                self.logger.info(f"{e.__class__.__name__}: {e}\n")
                self.logger.info("Traceback:\n{}\n".format(traceback.format_exc()))
                messages.append(
                    SystemMessage(
                        content=f"Your response resulted in the following error:\n{e.__class__.__name__}: {e}\n{traceback.format_exc()}\n\nPlease correct your response to prevent this error."
                    )
                )
        # Auto-training
        if user_input is None or user_input.strip() == "":
            user_input = self._generate_question(self.analysis_guide)
        if (
            auto_train
            and self.analysis_output is not None
            and len(self.analysis_output) > 0
            and self.analysis_guide is not None
            and self.analysis_guide.strip() != ""
        ):
            self.logger.info("Saving data for next training round\n")
            self.vector_store.add_training_plan(
                question=user_input, sql=self.analysis_guide
            )
        return self.analysis_output

    def run_analysis_for_plotting(self, user_input: str, **kwargs):
        start_time = time.time()
        question_sql_list = self.vector_store.get_similar_question_sql(
            user_input, **kwargs
        )
        ddl_list = self.vector_store.get_related_ddl(user_input, **kwargs)
        doc_list = self.vector_store.get_related_documentation(user_input, **kwargs)
        messages = self._get_sql_prompt(
            user_input=user_input,
            question_sql_list=question_sql_list,
            ddl_list=ddl_list,
            doc_list=doc_list,
            for_plotting=True,
        )
        for _ in range(3):
            try:
                sql_query = extract_sql(
                    self.model.run(messages=messages, **kwargs).message.content,
                    self.logger,
                )
                plot_df = self.connector.run_sql(sql_query)
                return plot_df
            except RecursionError:
                raise RecursionError(
                    "The request could not be completed. Please wait a while and try again."
                )
            except Exception as e:
                if time.time() - start_time > 30:
                    raise TimeoutError(
                        "The request could not be completed. Please wait a while and try again."
                    )
                self.logger.info(f"{e.__class__.__name__}: {e}\n")
                self.logger.info("Traceback:\n{}\n".format(traceback.format_exc()))
                messages.append(
                    SystemMessage(
                        content=f"Your response resulted in the following error:\n{e.__class__.__name__}: {e}\n{traceback.format_exc()}\n\nPlease correct your response to prevent this error."
                    )
                )

    def _generate_sql(
        self, user_input: str, for_plotting: bool = False, **kwargs
    ) -> str:
        question_sql_list = self.vector_store.get_similar_question_sql(
            user_input, **kwargs
        )
        ddl_list = self.vector_store.get_related_ddl(user_input, **kwargs)
        doc_list = self.vector_store.get_related_documentation(user_input, **kwargs)
        # get response from LLM in llm_response
        llm_response = self.model.run(
            messages=self._get_sql_prompt(
                user_input=user_input,
                question_sql_list=question_sql_list,
                ddl_list=ddl_list,
                doc_list=doc_list,
                for_plotting=for_plotting,
            ),
            **kwargs,
        ).message.content
        return llm_response

    def _generate_question(self, sql: str, **kwargs) -> str:
        kwargs["max_tokens"] = 500
        output = self.model.run(
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
