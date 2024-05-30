"""
PlotFactory class - generate visualizations based on user input and data sources.
"""

# standard library imports
import os
import re
import time
import logging
import warnings
import traceback

# third-party imports
import numpy as np
import pandas as pd
import seaborn as sns
import pmdarima as pm
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt

# local imports
from lyzr.base.llm import LiteLLM
from lyzr.base.prompt import LyzrPromptFactory
from lyzr.base.base import UserMessage, SystemMessage
from lyzr.data_analyzr.models import FactoryBaseClass
from lyzr.data_analyzr.utils import deterministic_uuid
from lyzr.data_analyzr.analysis_handler.utils import (
    handle_plotpath,
    extract_df_names,
    iterate_llm_calls,
    make_locals_string,
    extract_python_code,
    extract_column_names,
    remove_print_and_plt_show,
)
from lyzr.data_analyzr.db_connector import DatabaseConnector
from lyzr.data_analyzr.vector_store_utils import ChromaDBVectorStore

default_plot_path = "generated_plots/plot.png"


class PlotFactory(FactoryBaseClass):
    """
    PlotFactory is a class responsible for generating visual representations based on user input.

    This class leverages a language model (LLM) to interpret user queries and generate corresponding plots.
    It supports various data sources, including database connectors and dictionaries of pandas DataFrames.
    The class also manages retries, time limits, and automatic training of the vector store with new plotting examples.

    Attributes:
        plotting_library (str): The plotting library used, default is "matplotlib".
        connector (DatabaseConnector): The database connector for fetching data, if provided.
        df_dict (dict): Dictionary of pandas DataFrames, if provided.
        analysis_output (pd.DataFrame or dict): The output of analysis, if provided.
        plot_path (str): Path where the plot will be saved.
        locals_ (dict): Dictionary of local variables for use in plotting functions.
        code (str): The executed plotting code.
        fig (matplotlib.figure.Figure): The generated plot figure.

    Methods:
        generate_output(user_input: str, plot_path: str = None, **kwargs) -> str:
            Generates a visual representation based on the provided user input.

        get_prompt_messages(user_input: str) -> list:
            Generate a list of prompt messages based on the user's input.

        _get_message_sections_and_dict(user_input: str) -> tuple[list, dict]:
            Extracts and organizes message sections and a corresponding dictionary based on user input.

        _get_message_docs(user_input: str) -> tuple[str, set]:
            Retrieve related documentation and dataframe names using the user input.

        _get_locals() -> dict:
            Retrieve a dictionary of local variables for use in plotting functions.

        _add_sql_examples(user_input: str, system_message_sections: list, system_message_dict: dict):
            Add SQL examples to the system message sections and dictionary based on user input.

        extract_and_execute_code(llm_response: str):
            Executes the plotting code extracted from the provided LLM response.

        save_plot_image() -> str:
            Saves the current plot to a file specified by `self.plot_path`.

        auto_train(user_input: str, code: str, **kwargs):
            Automatically trains the vector store with a given user input and corresponding plot code.
    """

    def __init__(
        self,
        llm: LiteLLM,
        logger: logging.Logger,
        context: str,
        data_kwargs: dict,
        vector_store: ChromaDBVectorStore,
        max_retries: int = None,
        time_limit: int = None,
        auto_train: bool = None,
        **llm_kwargs,  # model_kwargs: dict
    ):
        """
        Initializes the PlotFactory instance.

        Args:
            llm (LiteLLM): The llm instance to be used.
            logger (logging.Logger): Logger instance for logging purposes.
            context (str): The context for the given query. If empty, pass "".
            data_kwargs (dict): Dictionary containing data-related parameters.
                - connector (DatabaseConnector, optional): Database connector for fetching data.
                - df_dict (dict, optional): Dictionary of pandas DataFrames.
                - analysis_output (optional): Output of analysis.
            vector_store (ChromaDBVectorStore): The vector store for managing related queries and database documentation.
            max_retries (int, optional): Maximum number of retries for plotting. Defaults to 10 if not provided.
            time_limit (int, optional): Time limit for plotting in seconds. Defaults to 60 if not provided.
            auto_train (bool, optional): Whether to automatically add to training data. Defaults to True.
            **llm_kwargs: Additional keyword arguments for the language model.

        Raises:
            ValueError: If neither `connector` nor `df_dict` is provided.
            ValueError: If both `connector` and `df_dict` are provided.
            ValueError: If `connector` is not a DatabaseConnector instance or `df_dict` is not a dictionary of pandas DataFrames.

        Example:
            from lyzr.base.llm import LiteLLM
            from lyzr.data_analyzr.vector_store_utils import ChromaDBVectorStore
            from lyzr.data_analyzr.analysis_handler import PlotFactory
            llm = LiteLLM.from_defaults(model="gpt-4o")
            logger = logging.getLogger('plot_logger')
            context = "Provide context for the plot"
            data_kwargs = {"df_dict": {"example_df": pd.DataFrame(...)}}
            vector_store = ChromaDBVectorStore(path="path/to/vector_store")
            plotter = PlotFactory(
                llm=llm,
                logger=logger,
                context=context,
                data_kwargs=data_kwargs,
                vector_store=vector_store,
                max_retries=10, # default
                time_limit=60, # default
                auto_train=True
            )
        """
        super().__init__(
            llm=llm,
            logger=logger,
            context=context,
            vector_store=vector_store,
            max_retries=10 if max_retries is None else max_retries,
            time_limit=60 if time_limit is None else time_limit,
            auto_train=auto_train,
            llm_kwargs=llm_kwargs,
        )
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

    def generate_output(self, user_input: str, plot_path: str = None, **kwargs) -> str:
        """
        Generates a visual representation based on the provided user input.

        This is the primary method for generating plots.
        - Processes the user input to generate visualisation.
        - Handles retries and time limits for the code extraction and execution process.
        - Uses iterate_llm_calls to handle retries and time limits.

        Args:
            user_input (str): The input string from the user that describes the desired plot.
            plot_path (str): Path where the plot will be saved.
            **kwargs: Additional keyword arguments to customize the behavior of the method.
                - max_retries (int): Maximum number of retries for LLM calls. Defaults to self.params.max_retries.
                - time_limit (int): Time limit for LLM calls. Defaults to self.params.time_limit.
                - auto_train (bool): Whether to automatically add the generated code to the training data. Defaults to self.params.auto_train.

        Returns:
            str: The file path of the saved plot image, or an empty string if no plot was generated.

        Example:
            saved_plot_path = plotter.get_visualisation("Plot a bar chart of sales by region")
            from PIL import Image
            Image.open(plot_path).show()
        """
        self.plot_path = handle_plotpath(
            plot_path=plot_path,
            output_format="png",
            uuid=deterministic_uuid([user_input, str(time.time())]),
            logger=self.logger,
        )
        messages = self.get_prompt_messages(user_input)
        self.extract_and_execute_code = iterate_llm_calls(
            max_retries=kwargs.pop("max_retries", self.params.max_retries),
            llm=self.llm,
            llm_messages=messages,
            logger=self.logger,
            log_messages={
                "start": f"Generating plot for query: {user_input}",
                "end": f"Finished generating plot for query: {user_input}",
            },
            time_limit=kwargs.pop("time_limit", self.params.time_limit),
            llm_kwargs=dict(
                max_tokens=2000,
                top_p=1,
            ),
        )(self.extract_and_execute_code)
        self.fig = self.extract_and_execute_code()
        if self.fig is None:
            plt.close("all")
            return ""
        self.auto_train(user_input, self.code, **kwargs)
        return self.save_plot_image()

    def get_prompt_messages(self, user_input: str) -> list:
        """
        Generate a list of prompt messages based on the user's input.

        This method processes the user's input to create a series of messages that
        can be used to generate a plot. It retrieves message sections and a dictionary
        of system messages, fetches similar plotting code examples from the vector
        store, and appends them to the message list. Finally, it constructs and returns
        a list of messages, including the user's input.

        Args:
            user_input (str): The input provided by the user.

        Returns:
            list: A list of messages including system prompt, examples, and user input.
        """
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
        """
        Extracts and organizes message sections and a corresponding dictionary based on user input.

        Args:
            user_input (str): The input provided by the user.

        Returns:
            tuple[list, dict]: A tuple containing a list of message sections and a dictionary with context and data.

        Procedure:
            - Get related documentation and DataFrame names from _get_message_docs.
            - Get local variables from _get_locals.
            - If a DatabaseConnector is present, add SQL plot sections and examples.
            - If a DataFrame dictionary is present, add Python plot sections and examples.
            - Return the message sections and dictionary.
        """
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
        """
        Retrieve related documentation and dataframe names using the user input.

        Args:
            user_input (str): The input string provided by the user to search for related documentation.

        Returns:
            tuple[str, set]: A tuple containing:
                - A concatenated string of related documentation if any are found, otherwise None.
                - A set of dataframe names extracted from the documentation if any are found, otherwise None.

        Procedure:
            - Get related documentation from the vector store.
            - Extract dataframe names from the documentation.
            - Return the concatenated documentation string and the set of dataframe names.
        """
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
        """
        Retrieve a dictionary of local variables for use in plotting functions.

        Returns:
            dict: A dictionary with keys as module names and values as the corresponding
            imported modules or DataFrames from `analysis_output`.

        Procedure:
            - Construct a dictionary containing commonly used libraries and modules.
            - Include the `analysis_output` attribute if it is a pandas DataFrame.
            - Update the dictionary with items from `analysis_output` if it is a dictionary containing DataFrames.
            - Return the dictionary.
        """
        locals_ = {
            "pd": pd,
            "np": np,
            "pm": pm,
            "sm": sm,
            "stats": stats,
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
        """
        Add SQL examples to the system message sections and dictionary based on user input.

        Args:
            user_input (str): The input provided by the user.
            system_message_sections (list): A list of sections in the system message.
            system_message_dict (dict): A dictionary containing the system message content.

        Returns:
            tuple: Updated `system_message_sections` and `system_message_dict` with SQL examples included if any were found.

        Procedure:
            - Retrieve SQL examples similar to the user's input from the vector store.
            - If any examples are found, append them to the system message sections and format them into a string.
            - Add the formatted string to the system message dictionary under the key "sql_examples".
            - Return the updated system message sections and dictionary.
        """
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

    def extract_and_execute_code(self, llm_response: str):
        """
        Executes the plotting code extracted from the provided LLM response.
        To be used as a callback function for iterate_llm_calls.

        Args:
            llm_response (str): The response from the language model containing the plotting code.

        Returns:
            matplotlib.figure.Figure: The generated plot figure.

        Raises:
            AssertionError: When `self.connector` is not an instance of DatabaseConnector and:
                - `df_dict` is not a dictionary
                - if `df_dict` contains non-pandas DataFrame objects.

        Logs:
            - Info: Extracted Python code.

        Procedure:
            - Extracts Python code from the LLM response and removes any `print`, `plt.show()`, `plt.savefig()` statements.
            - Logs the extracted code.
            - If a DatabaseConnector is not present:
                a. Identifies dataframe names used in the code and ensures they exist in `df_dict`.
                b. For each identified dataframe, drops rows with NaN values in the columns used in the code
                and updates the local scope.
            - Suppresses pandas chained assignment warnings and other warnings.
            - Executes the processed code within the local scope.
            - Stores the executed code in the `code` attribute.
            - Returns the value of the 'result' variable from the local scope.
        """
        code = remove_print_and_plt_show(extract_python_code(llm_response))
        self.logger.info(f"Extracted Python code:\n{code}")
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
        pd.options.mode.chained_assignment = None
        warnings.filterwarnings("ignore")
        globals_ = self.locals_
        exec(code, globals_, self.locals_)
        self.code = code
        return self.locals_["fig"]

    def save_plot_image(self) -> str:
        """
        Saves the current plot to a file specified by `self.plot_path`.

        Returns:
            str: The path where the plot was saved or intended to be saved.

        Logs:
            - Info: When the plot is successfully saved.
            - Error: When there is an issue saving the plot, including the traceback for debugging.
        """
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

    def auto_train(self, user_input: str, code: str, **kwargs):
        """
        Automatically trains the vector store with a given user input and corresponding plot code.

        Args:
            user_input (str): The input provided by the user, typically a question or query.
            code (str): The plot code or script that corresponds to the user input.
            auto_train (bool): Whether to automatically add the analysis to the training data,
                defaults to self.params.auto_train.

        Procedure:
            - Checks if the `auto_train` keyword is set to True.
            - If the plot code is not empty, adds the user input and the plot code to the vector store.
        """
        if (
            kwargs.pop("auto_train", self.params.auto_train)
            and code is not None
            and code.strip() != ""
        ):
            self.vector_store.add_training_plan(question=user_input, plot_code=code)
