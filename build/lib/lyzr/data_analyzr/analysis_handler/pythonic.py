"""
PythonicAnalysisFactory class - generate and execute Pythonic code from text inputs.
"""

# standard library imports
import re
import logging
import warnings
from typing import Union

# third-party imports
import numpy as np
import pandas as pd
import pmdarima as pm
from scipy import stats
import statsmodels.api as sm

# local imports
from lyzr.base.llm import LiteLLM
from lyzr.base.errors import DependencyError
from lyzr.base.prompt import LyzrPromptFactory
from lyzr.base.base import UserMessage, SystemMessage
from lyzr.data_analyzr.models import FactoryBaseClass
from lyzr.data_analyzr.analysis_handler.utils import (
    extract_df_names,
    iterate_llm_calls,
    make_locals_string,
    extract_python_code,
    extract_column_names,
    handle_analysis_output,
    remove_print_and_plt_show,
)
from lyzr.data_analyzr.vector_store_utils import ChromaDBVectorStore


class PythonicAnalysisFactory(FactoryBaseClass):
    """
    PythonicAnalysisFactory is a class for generating and executing Pythonic analysis based on user input.

    Attributes:
        df_dict (dict): Dictionary containing dataframes for analysis.

    Methods:
        __init__(llm, logger, context, df_dict, vector_store, max_retries=None, time_limit=None, auto_train=None, **llm_kwargs):
            Initializes a PythonicAnalysisFactory instance.
        generate_output(user_input, **kwargs):
            Runs analysis and generates output based on the provided user input.
        get_prompt_messages(user_input):
            Generates a list of prompt messages based on the user's input.
        get_analysis_guide(user_input):
            Generates an analysis guide based on the user's input.
        _get_locals_and_docs(system_message_sections, system_message_dict, user_input):
            Retrieves local variables and related documentation based on user input.
        extract_and_execute_code(llm_response):
            Extracts Python code from a given LLM response, processes it, and executes it within a controlled environment.
        auto_train(user_input, code, **kwargs):
            Adds the user input and generated Python code to the vector store if the auto_train flag is set.
    """

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
        **llm_kwargs,
    ):
        """
        Initialize a PythonicAnalysisFactory instance.

        Args:
            llm (LiteLLM): The llm instance to be used.
            logger (logging.Logger): Logger instance for logging information.
            context (str): The context for the given query. If empty, pass "".
            df_dict (dict): Dictionary containing dataframes for analysis.
            vector_store (ChromaDBVectorStore): The vector store for managing related queries and database documentation.
            max_retries (int, optional): Maximum number of retries for analysis. Defaults to 10 if not provided.
            time_limit (int, optional): Time limit for analysis in seconds. Defaults to 45 if not provided.
            auto_train (bool, optional): Whether to automatically add to training data. Defaults to True.
            **llm_kwargs: Additional keyword arguments for the llm.

        Example:
            from lyzr.base.llm import LiteLLM
            from lyzr.data_analyzr.vector_store_utils import ChromaDBVectorStore
            from lyzr.data_analyzr.analysis_handler import PythonicAnalysisFactory
            import logging
            import pandas as pd

            llm = LiteLLM.from_defaults(model="gpt-4o")
            logger = logging.getLogger(__name__)
            context = "Context for the analysis."
            df_dict = {"df": pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})}
            vector_store = ChromaDBVectorStore(path="path/to/vector_store", logger=logger)
            analysis_factory = PythonicAnalysisFactory(
                llm=llm,
                logger=logger,
                context=context,
                df_dict=df_dict,
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
        self.df_dict = df_dict

    def generate_output(
        self, user_input: str, **kwargs
    ) -> Union[str, pd.DataFrame, dict[str, pd.DataFrame], None]:
        """
        Run analysis and generate output based on the provided user input.

        This is the primary method for generating and executing Pythonic analysis.
        - Processes the user input to generate Pythonic analysis output.
        - Handles retries and time limits for the code extraction and execution process.
        - Uses iterate_llm_calls to handle retries and time limits.

        Example:
            output = analysis_factory.generate_output("Your question here.")
            print(output)

        Args:
            user_input (str): The input string provided by the user for analysis.
            **kwargs: Additional keyword arguments to customize the analysis process.
                - max_retries (int): Maximum number of retries for LLM calls. Defaults to the value in self.params.max_retries.
                - time_limit (int): Time limit for the analysis execution. Defaults to the value in self.params.time_limit.
                - auto_train (bool): Whether to automatically add the analysis to training data. Defaults to the value in self.params.auto_train.

        Returns:
            Union[str, pd.DataFrame, dict[str, pd.DataFrame], None]: The output of the analysis.

        Raises:
            DependencyError: If the user input is None or an empty string.
        """
        user_input = user_input.strip()
        if user_input is None or user_input == "":
            raise DependencyError("A user input is required for analysis.")
        self.extract_and_execute_code = iterate_llm_calls(
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
        )(self.extract_and_execute_code)
        self.output = handle_analysis_output(self.extract_and_execute_code())
        self.auto_train(user_input, self.code, **kwargs)
        return self.output

    def get_prompt_messages(self, user_input: str) -> list:
        """
        Generate a list of prompt messages based on the user's input.

        This method constructs a series of messages to be used with the LLM.
        - Incorporates context and examples relevant to the user's input.
        - List of messages includes system messages with context, guides, local variables,
        documentation, and historical examples, followed by the user's input.

        Args:
            user_input (str): The input provided by the user.

        Returns:
            list[ChatMessage]: A list of chat messages generated based on the user's input.
        """
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
        question_examples_list = self.vector_store.get_related_python_code(user_input)
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

    def get_analysis_guide(self, user_input: str) -> str:
        """
        Generate an analysis guide based on the user's input.

        Args:
            user_input (str): The input provided by the user for which the analysis guide is to be generated.

        Returns:
            str: The content of the analysis guide generated by the language model.

        Procedure:
            - Define system message sections and a dictionary to store system message format strings.
            - Retrieve related documentation from the vector store based on the user input.
            - Add the documentation to the system message dictionary.
            - Generate an analysis guide using the llm based on the system message sections and dictionary.
        """
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

    def _get_locals_and_docs(
        self, system_message_sections: list, system_message_dict: dict, user_input: str
    ) -> tuple[list, dict]:
        """
        Retrieve local variables and related documentation based on user input.

        Args:
            system_message_sections (list): A list to append sections of system messages.
            system_message_dict (dict): A dictionary to store system message format strings.
            user_input (str): The user input string to search for related documentation.

        Returns:
            tuple[list, dict]: Updated system message sections and system message dictionary.

        Procedure:
            - Retrieve related documentation from the vector store based on the user input.
            - Extract dataframe names from the documentation and ensure they exist in the local scope.
            - Update the local scope with dataframes that have been cleaned of NaN values.
            - Update the system message dictionary with the local variables and related documentation.
        """
        self.locals_ = {
            "pd": pd,
            "np": np,
            "pm": pm,
            "sm": sm,
            "stats": stats,
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

    def extract_and_execute_code(self, llm_response: str):
        """
        Extracts Python code from a given LLM response, processes it, and executes it within a controlled environment.
        To be used as a callback function for iterate_llm_calls.

        Args:
            llm_response (str): The response from the language model containing Python code.

        Returns:
            Any: The result of the executed code, expected to be stored in the 'result' variable within the executed code's local scope.

        Raises:
            AssertionError: If any of the dataframes in `df_dict` are not instances of `pd.DataFrame`.

        Logs:
            - Info: The extracted Python code.

        Procedure:
            - Extracts Python code from the LLM response and removes any `print`, `plt.show()`, `plt.savefig()` statements.
            - Logs the extracted code.
            - Identifies dataframe names used in the code and ensures they exist in `df_dict`.
            - For each identified dataframe, drops rows with NaN values in the columns used in the code and updates the local scope.
            - Suppresses pandas chained assignment warnings and other warnings.
            - Executes the processed code within the local scope.
            - Stores the executed code in the `code` attribute.
            - Returns the value of the 'result' variable from the local scope.
        """
        code = remove_print_and_plt_show(extract_python_code(llm_response))
        self.logger.info(f"Extracted Python code:\n{code}")
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
        globals_ = self.locals_
        exec(code, globals_, self.locals_)
        self.code = code
        return self.locals_["result"]

    def auto_train(self, user_input: str, code: str, **kwargs):
        """
        Adds the user input and generated Python code to the vector store if the auto_train flag is set.

        Args:
            user_input (str): The input provided by the user to be used for training.
            code (str): The Python code generated by the analysis to be used for training.
            auto_train (bool): Optional flag to control whether the training should be performed. Defaults to the instance's 'auto_train' parameter.

        Procedure:
            - Checks if the 'auto_train' flag is set either in the keyword arguments or in the instance parameters.
            - Checks if the 'output' and 'code' attributes are not None and the 'code' attribute is not an empty string.
            - Adds a training plan to the vector store using the provided user input and the current Python code.
        """
        if (
            kwargs.pop("auto_train", self.params.auto_train)
            and self.output is not None
            and code is not None
            and code.strip() != ""
        ):
            self.vector_store.add_training_data(question=user_input, python_code=code)
