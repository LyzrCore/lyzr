# standard library imports
import re
import logging
import warnings
from typing import Union

# third-party imports
import numpy as np
import pandas as pd

# local imports
from lyzr.base.llm import LiteLLM
from lyzr.base.errors import DependencyError
from lyzr.base.prompt import LyzrPromptFactory
from lyzr.data_analyzr.utils import iterate_llm_calls
from lyzr.base.base import UserMessage, SystemMessage
from lyzr.data_analyzr.models import FactoryBaseClass
from lyzr.data_analyzr.analysis_handler.utils import (
    extract_df_names,
    make_locals_string,
    extract_python_code,
    extract_column_names,
    handle_analysis_output,
    remove_print_and_plt_show,
)
from lyzr.data_analyzr.vector_store_utils import ChromaDBVectorStore

pd.options.mode.chained_assignment = None
warnings.filterwarnings("ignore")


class PythonicAnalysisFactory(FactoryBaseClass):

    def __init__(
        self,
        llm: LiteLLM,
        logger: logging.Logger,
        context: str,
        df_dict: dict,
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
            time_limit=45 if time_limit is None else time_limit,
            auto_train=auto_train,
            llm_kwargs=llm_kwargs,
        )
        self.df_dict = df_dict
        assert isinstance(self.df_dict, dict), "df_dict must be a dictionary"

    def run_analysis(
        self, user_input: str, **kwargs
    ) -> Union[str, pd.DataFrame, dict[str, pd.DataFrame], None]:
        user_input = user_input.strip()
        self.locals_ = {}
        if user_input is None or user_input == "":
            raise DependencyError("A user input is required for analysis.")
        self.execute_analysis_code = iterate_llm_calls(
            max_retries=kwargs.pop("max_retries", self.params.max_retries),
            llm=self.llm,
            llm_messages=self.get_prompt_messages(user_input),
            logger=self.logger,
            log_messages={
                "start": f"Starting Pythonic analysis for query: {user_input}",
                "end": f"Ending Pythonic analysis for query: {user_input}",
            },
            time_limit=kwargs.pop("time_limit", self.params.time_limit),
            llm_kwargs=dict(
                max_tokens=2000,
                top_p=1,
            ),
        )(self.execute_analysis_code)
        self.output = handle_analysis_output(self.execute_analysis_code())
        if kwargs.pop("auto_train", self.params.auto_train) and self.output is not None:
            self.add_training_data(user_input, self.code)
        return self.output

    def get_prompt_messages(self, user_input: str) -> list:
        system_message_sections = [
            "context",
            "external_context",
            "task",
            "closing",
        ]
        system_message_dict = {"context": self.context}
        # add analysis guide
        self.guide = self.get_analysis_guide(user_input)
        if self.guide is not None and self.guide != "":
            system_message_sections.append("guide")
            system_message_dict["guide"] = self.guide
        # add locals and docs
        system_message_sections, system_message_dict = self._get_locals_and_docs(
            system_message_sections=system_message_sections,
            system_message_dict=system_message_dict,
            user_input=user_input,
        )
        # add question examples
        question_examples_list = self.vector_store.get_similar_python_code(user_input)
        if len(question_examples_list) > 0:
            system_message_sections.append("history")
        messages = [
            LyzrPromptFactory(name="analysis_code", prompt_type="system").get_message(
                use_sections=system_message_sections,
                **system_message_dict,
            )
        ]
        for example in question_examples_list:
            if (
                (example is not None)
                and ("question" in example)
                and ("python_code" in example)
            ):
                messages.append(UserMessage(content=example["question"]))
                messages.append(SystemMessage(content=example["python_code"]))
        messages.append(UserMessage(content=user_input))
        return messages

    def _get_locals_and_docs(
        self, system_message_sections: list, system_message_dict: dict, user_input: str
    ) -> tuple[list, dict]:
        self.locals_ = {
            "pd": pd,
            "np": np,
        }
        system_message_sections.append("locals")
        doc_list = self.vector_store.get_related_documentation(user_input)
        df_names = set()
        if len(doc_list) > 0:
            system_message_sections.append("doc_addition_text")
            doc_str = ""
            for doc_item in doc_list:
                doc_str += f"{doc_item}\n"
            for name in re.finditer(r"(\w+) dataframe", doc_str, re.IGNORECASE):
                df_names.update(name.groups())
            system_message_dict["doc"] = doc_str
        for name, df in self.df_dict.items():
            if name in df_names:
                self.locals_[name] = df
        system_message_dict["locals"] = make_locals_string(self.locals_)
        return system_message_sections, system_message_dict

    def execute_analysis_code(self, llm_response: str):
        code = remove_print_and_plt_show(
            extract_python_code(llm_response, logger=self.logger)
        )
        df_names = extract_df_names(code, list(self.df_dict.keys()))
        for name in df_names:
            df = self.df_dict[name]
            assert isinstance(
                df, pd.DataFrame
            ), "df_dict must must only contain pandas DataFrames"
            columns = extract_column_names(code, self.df_dict[name])
            self.locals_[name] = df.dropna(subset=columns)
        pd.options.mode.chained_assignment = None
        warnings.filterwarnings("ignore")
        exec(code, globals(), self.locals_)
        self.code = code
        return self.locals_["result"]

    def add_training_data(self, user_input: str, code: str):
        if code is not None and code.strip() != "":
            self.vector_store.add_training_plan(question=user_input, python_code=code)

    def get_analysis_guide(self, user_input: str) -> str:
        system_message_sections = [
            "context",
            "external_context",
            "task",
        ]
        system_message_dict = {"context": self.context}
        doc_list = self.vector_store.get_related_documentation(user_input)
        if len(doc_list) > 0:
            system_message_sections.append("doc_addition_text")
            system_message_dict["doc"] = ""
            for doc_item in doc_list:
                system_message_dict["doc"] += f"{doc_item}\n"
        messages = [
            LyzrPromptFactory("ml_analysis_guide", "system").get_message(
                use_sections=system_message_sections,
                **system_message_dict,
            ),
        ]
        llm_response = self.llm.run(messages=messages)
        return llm_response.message.content.strip()
