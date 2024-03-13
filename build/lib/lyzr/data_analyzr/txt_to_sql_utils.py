# standard library imports
import logging

# local imports
from lyzr.base.prompt import Prompt
from lyzr.base.errors import MissingValueError
from lyzr.base.llms import LLM, set_model_params
from lyzr.data_analyzr.output_handler import extract_sql
from lyzr.data_analyzr.db_connector import DatabaseConnector
from lyzr.data_analyzr.vector_store_utils import ChromaDBVectorStore


class TxttoSQLFactory:

    def __init__(
        self,
        model: LLM,
        db_connector: DatabaseConnector,
        logger: logging.Logger,
        context: str,
        vector_store: ChromaDBVectorStore,
        model_kwargs: dict = None,
    ):
        self.model = model
        self.model_kwargs = set_model_params(
            {"seed": 123, "temperature": 0.1, "top_p": 0.5}, model_kwargs or {}
        )
        self.context = context
        self.logger = logger

        self.connector = db_connector
        self.vector_store = vector_store
        if self.vector_store is None:
            raise MissingValueError("vector_store")

        self.analysis_output, self.analysis_guide = None, None

    def get_analysis_output(
        self,
        user_input: str,
        auto_train: bool = True,
    ):
        # SQL Generation
        self.logger.info("Generating SQL Query\n")
        self.analysis_guide = self._generate_sql(user_input=user_input)
        sql_query = extract_sql(self.analysis_guide, self.logger)
        self.logger.info(f"SQL Query generated:\n{sql_query}\n")

        # SQL Execution
        self.analysis_output = self.connector.run_sql(sql_query)
        self.logger.info(f"Analysis Output:\n{self.analysis_output}\n")

        # Auto-training
        if user_input is None or user_input.strip() == "":
            user_input = self._generate_question(sql_query)
        if (
            auto_train
            and self.analysis_output is not None
            and len(self.analysis_output) > 0
            and sql_query is not None
            and sql_query.strip() != ""
        ):
            self.logger.info("Saving data for next training round\n")
            self.vector_store.add_training_plan(question=user_input, sql=sql_query)

        return self.analysis_output

    def _generate_sql(self, user_input: str, **kwargs) -> str:
        question_sql_list = self.vector_store.get_similar_question_sql(
            user_input, **kwargs
        )
        ddl_list = self.vector_store.get_related_ddl(user_input, **kwargs)
        doc_list = self.vector_store.get_related_documentation(user_input, **kwargs)

        # get response from LLM in llm_response
        self.model.set_messages(
            messages=self._get_sql_prompt(
                user_input=user_input,
                question_sql_list=question_sql_list,
                ddl_list=ddl_list,
                doc_list=doc_list,
            )
        )
        kwargs = kwargs or {}
        self.model_kwargs = set_model_params(kwargs, self.model_kwargs, True)
        llm_response = self.model.run(**self.model_kwargs)
        return llm_response.choices[0].message.content

    def _generate_question(self, sql: str, **kwargs) -> str:
        self.model.set_messages(
            messages=[
                {
                    "role": "system",
                    "content": Prompt("sql_question_gen_pt").text,
                },
                {
                    "role": "user",
                    "content": sql,
                },
            ]
        )
        kwargs = kwargs or {}
        self.model_kwargs = set_model_params(kwargs, self.model_kwargs)
        self.model_kwargs = set_model_params(
            {"max_tokens": 500}, self.model_kwargs, True
        )
        output = self.model.run(**self.model_kwargs)
        return output.choices[0].message.content

    def _get_sql_prompt(
        self,
        user_input: str,
        question_sql_list: list,
        ddl_list: list,
        doc_list: list,
    ) -> list[dict]:
        prompt_text = Prompt("sql_prompt_pt").text
        addition_text = "\nYou may use the following {item_type} as a reference for what tables might be available:\n"
        closing_text = " \nAlso use responses to past questions to guide you."

        if len(ddl_list) > 0:
            prompt_text += addition_text.format(item_type="DDL statements")
            for ddl in ddl_list:
                prompt_text += f"{ddl}\n"

        if len(doc_list) > 0:
            prompt_text += addition_text.format(item_type="documentation")
            for doc in doc_list:
                prompt_text += f"{doc}\n"

        prompt_text += closing_text
        messages = [{"role": "system", "content": prompt_text}]

        for example in question_sql_list:
            if example is not None:
                if "question" in example and "sql" in example:
                    messages.append({"role": "user", "content": example["question"]})
                    messages.append({"role": "system", "content": example["sql"]})

        messages.append({"role": "user", "content": user_input})
        return messages
