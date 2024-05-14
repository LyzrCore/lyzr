# standard library imports
import json
import logging

# third-party imports
import pandas as pd

# local imports
from lyzr.base.llm import LiteLLM
from lyzr.data_analyzr.utils import (
    get_columns_names,
    format_df_with_info,
    iterate_llm_calls,
)
from lyzr.base.errors import DependencyError
from lyzr.base.prompt import LyzrPromptFactory
from lyzr.data_analyzr.models import FactoryBaseClass
from lyzr.base.base import ChatMessage, UserMessage, SystemMessage
from lyzr.data_analyzr.vector_store_utils import ChromaDBVectorStore
from lyzr.data_analyzr.analysis_handler.utils import AnalysisExecutor
from lyzr.data_analyzr.analysis_handler.models import PythonicAnalysisModel


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
            max_retries=max_retries,
            time_limit=time_limit,
            auto_train=auto_train,
            llm_kwargs=llm_kwargs,
        )
        self.df_dict = df_dict

    def run_analysis(self, user_input: str, **kwargs) -> pd.DataFrame:
        user_input = user_input.strip()
        if user_input is None or user_input == "":
            raise DependencyError("A user input is required for analysis.")
        for_plotting = kwargs.pop("for_plotting", False)
        self.analysis_guide = self.get_analysis_guide(
            user_input, for_plotting=for_plotting
        )
        self.steps = self.generate_analysis_steps(
            for_plotting=for_plotting, user_input=user_input, **kwargs
        )
        self.analysis_output = self.execute_analysis_steps()
        if (
            kwargs.pop("auto_train", self.params.auto_train)
            and self.analysis_output is not None
            and len(self.analysis_output) > 0
        ):
            self.add_training_data(user_input, json.dumps(self.steps.model_dump()))
        return self.analysis_output

    def get_analysis_guide(self, user_input: str, for_plotting: bool) -> str:
        if for_plotting:
            system_message_sections = [
                "context",
                "external_context",
                "plotting_task",
            ]
        else:
            system_message_sections = [
                "context",
                "external_context",
                "task",
            ]
        messages = [
            LyzrPromptFactory("ml_analysis_guide", "system").get_message(
                use_sections=system_message_sections,
                context=self.context,
            ),
            LyzrPromptFactory("ml_analysis_guide", "user").get_message(
                df_details=format_df_with_info(self.df_dict),
                question=user_input,
            ),
        ]
        llm_response = self.llm.run(messages=messages)
        return llm_response.message.content

    def generate_analysis_steps(
        self, for_plotting: bool, user_input: str, **kwargs
    ) -> PythonicAnalysisModel:
        messages = self.get_prompt_messages(user_input, for_plotting=for_plotting)
        self.get_analysis_steps = iterate_llm_calls(
            max_retries=kwargs.pop("max_retries", self.params.max_retries),
            llm=self.llm,
            llm_messages=messages,
            logger=self.logger,
            log_messages={
                "start": f"Starting Pythonic analysis for query: {user_input}",
                "end": f"Ending Pythonic analysis for query: {user_input}",
            },
            time_limit=kwargs.pop("time_limit", self.params.time_limit),
            llm_kwargs=dict(
                response_format={"type": "json_object"},
                max_tokens=2000,
                top_p=1,
            ),
        )(self.get_analysis_steps)
        steps = self.get_analysis_steps()
        return steps

    def get_prompt_messages(
        self, user_input: str, for_plotting: bool
    ) -> list[ChatMessage]:
        if for_plotting:
            system_message_sections = [
                "context",
                "external_context",
                "plotting_task",
                "closing",
            ]
        else:
            system_message_sections = [
                "context",
                "external_context",
                "task",
                "closing",
            ]
        schema = json.dumps(PythonicAnalysisModel.model_json_schema())
        messages = [
            LyzrPromptFactory(
                name="ml_analysis_steps", prompt_type="system"
            ).get_message(
                use_sections=system_message_sections,
                schema=schema,
                context=self.context,
            ),
            LyzrPromptFactory(name="ml_analysis_steps", prompt_type="user").get_message(
                df_details=format_df_with_info(self.df_dict),
                question=user_input,
                guide=self.analysis_guide,
            ),
        ]
        question_examples_list = self.vector_store.get_similar_analysis_steps(
            user_input
        )
        for example in question_examples_list:
            if (
                (example is not None)
                and ("question" in example)
                and ("steps" in example)
            ):
                messages.append(UserMessage(content=example["question"]))
                messages.append(SystemMessage(content=example["steps"]))
        return messages

    def get_analysis_steps(self, llm_response: str) -> PythonicAnalysisModel:
        steps = PythonicAnalysisModel(**json.loads(llm_response))
        return steps

    def execute_analysis_steps(self) -> pd.DataFrame:
        steps = self.steps
        if not isinstance(steps, PythonicAnalysisModel):
            steps = PythonicAnalysisModel(**json.loads(steps))
        df = self.df_dict[steps.analysis_df]
        for step in steps.steps:
            task_args = {
                key: value
                for key, value in step.args.model_dump().items()
                if key != "task"
            }
            analyzr = AnalysisExecutor(df=df, task=step.args.task, logger=self.logger)
            analyzr.func(**task_args)
            df = analyzr.df
            if isinstance(df, pd.Series):
                df = df.to_frame()
        df = df.loc[:, get_columns_names(df.columns, columns=steps.output_columns)]
        return df

    def add_training_data(self, user_input: str, steps: str):
        if steps is not None and steps.strip() != "":
            self.vector_store.add_training_plan(
                question=user_input, analysis_steps=steps
            )
