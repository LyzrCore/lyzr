# standard library imports
import os
import re
import logging
import traceback

# third-party imports
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# local imports
from lyzr.base.llm import LiteLLM
from lyzr.base.prompt import LyzrPromptFactory
from lyzr.data_analyzr.utils import iterate_llm_calls
from lyzr.base.base import UserMessage, SystemMessage
from lyzr.data_analyzr.models import FactoryBaseClass
from lyzr.data_analyzr.analysis_handler.utils import (
    handle_plotpath,
    extract_python_code,
    extract_sql,
    extract_df_names,
    make_locals_string,
    extract_column_names,
    remove_print_and_plt_show,
)
from lyzr.data_analyzr.db_connector import DatabaseConnector
from lyzr.data_analyzr.vector_store_utils import ChromaDBVectorStore

pd.options.mode.chained_assignment = None
default_plot_path = "generated_plots/plot.png"


class PlotFactory(FactoryBaseClass):

    def __init__(
        self,
        llm: LiteLLM,
        logger: logging.Logger,
        context: str,
        plot_path: str,
        data_kwargs: dict,
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
            time_limit=60 if time_limit is None else time_limit,
            auto_train=auto_train,
            llm_kwargs=llm_kwargs,
        )
        self.plot_path = handle_plotpath(plot_path, "png", self.logger)
        self.plotting_library = "matplotlib"
        self.connector = data_kwargs.get("connector", None)
        self.df_dict = data_kwargs.get("df_dict", None)
        self.analysis_output = data_kwargs.get("analysis_output", None)
        if self.connector is None and self.df_dict is None:
            raise ValueError(
                "Either connector or df_dict must be provided to make a plot."
            )
        if self.connector is not None and self.df_dict is not None:
            raise ValueError(
                "Both connector and df_dict cannot be provided to make a plot."
            )
        if not isinstance(self.connector, DatabaseConnector) and not isinstance(
            self.df_dict, dict
        ):
            raise ValueError(
                "Either connector or df_dict must be a DatabaseConnector or a dictionary of pandas DataFrames."
            )

    def get_visualisation(self, user_input: str, **kwargs) -> str:
        messages = self.get_prompt_messages(user_input)
        self.execute_plotting_code = iterate_llm_calls(
            max_retries=kwargs.pop("max_retries", self.params.max_retries),
            llm=self.llm,
            llm_messages=messages,
            logger=self.logger,
            log_messages={
                "start": f"Generating plot for query: {user_input}",
                "end": f"Finished generating plot for query: {user_input}",
            },
            time_limit=60,
            llm_kwargs=dict(
                max_tokens=2000,
                top_p=1,
            ),
        )(self.execute_plotting_code)
        self.fig = self.execute_plotting_code()
        if self.fig is None:
            plt.close("all")
            return ""
        if kwargs.pop("auto_train", self.params.auto_train):
            self.add_training_data(user_input, self.code)
        return self.save_plot_image()

    def get_prompt_messages(self, user_input: str) -> list:
        assert isinstance(
            self.vector_store, ChromaDBVectorStore
        ), "Vector store must be a ChromaDBVectorStore object."
        system_message_sections, system_message_dict = (
            self._get_message_sections_and_dict(user_input=user_input)
        )
        # add question examples
        question_examples_list = self.vector_store.get_similar_plotting_code(user_input)
        if len(question_examples_list) > 0:
            system_message_sections.append("history")
        messages = [
            LyzrPromptFactory(name="plotting_code", prompt_type="system").get_message(
                use_sections=system_message_sections,
                **system_message_dict,
            )
        ]
        for example in question_examples_list:
            if (
                (example is not None)
                and ("question" in example)
                and ("plot_code" in example)
            ):
                messages.append(UserMessage(content=example["question"]))
                messages.append(SystemMessage(content=example["plot_code"]))
        messages.append(UserMessage(content=user_input))
        return messages

    def _get_message_sections_and_dict(self, user_input: str) -> tuple[list, dict]:
        system_message_sections = ["context", "external_context"]
        system_message_dict = {"context": self.context}
        doc_str, df_names = self._get_message_docs(user_input)
        self.locals_ = self._get_locals()
        if self.connector is not None:
            system_message_sections.append("sql_plot")
            system_message_sections.append("locals")
            self.locals_["conn"] = self.connector
            if doc_str is not None:
                system_message_sections.append("doc_addition_text")
                system_message_dict["doc"] = doc_str
                system_message_dict["db_type"] = "sql database"
            system_message_sections, system_message_dict = self._add_sql_examples(
                user_input=user_input,
                system_message_sections=system_message_sections,
                system_message_dict=system_message_dict,
            )
        else:
            system_message_sections.append("python_plot")
            system_message_sections.append("locals")
            self.locals_.update(
                {name: df for name, df in self.df_dict.items() if name in df_names}
            )
            if doc_str is not None:
                system_message_sections.append("doc_addition_text")
                system_message_dict["doc"] = doc_str
                system_message_dict["db_type"] = "dataframe(s)"
        system_message_dict["locals"] = make_locals_string(self.locals_)
        return system_message_sections, system_message_dict

    def _get_message_docs(self, user_input: str) -> tuple[str, set]:
        doc_list = self.vector_store.get_related_documentation(user_input)
        df_names = set()
        if len(doc_list) > 0:
            doc_str = ""
            for doc_item in doc_list:
                doc_str += f"{doc_item}\n"
            for name in re.finditer(r"(\w+) dataframe", doc_str, re.IGNORECASE):
                df_names.update(name.groups())
            return doc_str, df_names
        return None, None

    def _get_locals(self):
        locals_ = {
            "pd": pd,
            "np": np,
            "plt": plt,
            "sns": sns,
        }
        if isinstance(self.analysis_output, pd.DataFrame):
            locals_["analysis_output"] = self.analysis_output
        elif isinstance(self.analysis_output, dict):
            locals_.update(
                {
                    name: df
                    for name, df in self.analysis_output.items()
                    if isinstance(df, pd.DataFrame)
                }
            )
        return locals_

    def _add_sql_examples(
        self, user_input: str, system_message_sections: list, system_message_dict: dict
    ):
        sql_examples = (
            self.vector_store.get_similar_question_sql(user_input)
            if isinstance(self.connector, DatabaseConnector)
            else []
        )
        if len(sql_examples) > 0:
            system_message_sections.append("sql_examples_text")
            sql_examples_str = ""
            for example in sql_examples:
                if example is not None:
                    if "question" in example and "sql" in example:
                        sql_examples_str += f"Question: {example['question']}\nSQL:\n{example['sql']}\n\n"
            system_message_dict["sql_examples"] = sql_examples_str
        return system_message_sections, system_message_dict

    def execute_plotting_code(self, llm_response: str):
        code = remove_print_and_plt_show(
            extract_python_code(llm_response, logger=self.logger)
        )
        if not isinstance(self.connector, DatabaseConnector):
            assert isinstance(self.df_dict, dict), "df_dict must be a dictionary."
            df_names = extract_df_names(code, list(self.df_dict.keys()))
            for name in df_names:
                df = self.df_dict[name]
                assert isinstance(
                    df, pd.DataFrame
                ), "df_dict must must only contain pandas DataFrames"
                columns = extract_column_names(code, self.df_dict[name])
                self.locals_[name] = df.dropna(subset=columns)
        exec(code, globals(), self.locals_)
        self.code = code
        return self.locals_["fig"]

    def extract_and_run_sql(self, llm_response: str):
        assert isinstance(
            self.connector, DatabaseConnector
        ), "Connector must be a DatabaseConnector object."
        sql_query = None
        sql_query = extract_sql(llm_response, logger=self.logger)
        self.locals_["df"] = self.connector.run_sql(sql_query)
        return sql_query

    def save_plot_image(self) -> str:
        plt.tight_layout()
        if not PlotFactory._savefig(self.plot_path):
            self.logger.error(
                f"Error saving plot at: {self.plot_path}. Plot not saved. Displaying plot instead. Access the plot using `.fig` attribute.",
                extra={
                    "function": "save_plot_image",
                    "traceback": traceback.format_exc().splitlines(),
                },
            )
            plt.show()
        else:
            self.logger.info(
                f"\nPlot saved at: {self.plot_path}\n",
                extra={"function": "save_plot_image"},
            )
            plt.close("all")
        return self.plot_path

    @staticmethod
    def _savefig(path: str):
        try:
            dir_path = os.path.dirname(path)
            if dir_path.strip() != "":
                os.makedirs(dir_path, exist_ok=True)
            plt.savefig(path)
            return True
        except Exception:
            if path == default_plot_path:
                return False
            PlotFactory._savefig(default_plot_path)
        return False

    def add_training_data(self, user_input: str, code: str):
        if code is not None and code.strip() != "":
            self.vector_store.add_training_plan(question=user_input, plot_code=code)
