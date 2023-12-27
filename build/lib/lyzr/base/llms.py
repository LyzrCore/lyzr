import os
from typing import Optional
from importlib import resources as impresources
from openai import OpenAI

from . import prompts


class Prompt:
    def __init__(self, prompt_name: str, prompt_text: Optional[str] = None):
        self.name = prompt_name
        self.text = prompt_text
        if self.text is None:
            # in-built prompt names end with _pt
            self.load_prompt()
            self.variables = self.get_variables()
        else:
            self.variables = self.get_variables()
            self.save_prompt()
    
    def get_variables(self):
        variables = []
        for word in self.text.split():
            if word.startswith("{") and word.endswith("}"):
                variables.append(word[1:-1])
        return variables
    
    def save_prompt(self):
        inp_file = impresources.files(prompts) / f"{self.name}.txt"
        with inp_file.open("w+") as f:
            f.write(self.text.encode("utf-8"))
    
    def load_prompt(self):
        try:
            inp_file = impresources.files(prompts) / f"{self.name}.txt"
            with inp_file.open("rb") as f:
                self.text = f.read().decode("utf-8")
        except FileNotFoundError:
            raise ValueError(
                f"No prompt with name '{self.name}' found. To use an in-built prompt, "
                "use one of the following prompt names: {get_prompts_list()}\n"
                "Or create a new prompt by passing the prompt text."
            )
    
    def edit_prompt(self, prompt_text: str):
        self.text = prompt_text
        self.variables = self.get_variables()
        self.save_prompt()
    
    def format(self, **kwargs):
        if self.text is None:
            raise ValueError(f"Please provide the text for the prompt '{self.name}'")
        prompt_text = self.text
        try:
            prompt_text = prompt_text.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Missing a value for variable: {e.args[0]}") from e
        return prompt_text


class LLM:
    def __init__(
        self,
        api_key: str,
        model_type: Optional[str] = "openai",
        model_name: Optional[str] = "gpt-3.5-turbo",
        prompt_name: Optional[str] = None,
        prompt: Optional[Prompt] = None,
        **kwargs,
    ):
        self.api_key = api_key
        self.model_type = model_type
        self.model_name = model_name
        self.voice = kwargs.get('voice')
        if prompt_name:
            self.prompt = Prompt(prompt_name)
        elif prompt:
            self.prompt = prompt
        else:
            self.prompt = None

        # Store any additional configuration not captured by existing properties.
        self.additional_params = kwargs
        
        # llm_params = {
        #     "temperature": 0,
        #     "top_p": 1,
        #     "presence_penalty": 0,
        #     "frequency_penalty": 0,
        #     "stop": None,
        #     "n": 1,
        #     "stream": False,
        #     "functions": None,
        #     "function_call": None,
        #     "logit_bias": None,
        #     "best_of": 1,
        #     "echo": False,
        #     "logprobs": None,
        #     "suffix": None,
        # }
        # for param in llm_params:
        #     setattr(self, param, kwargs.get(param, llm_params[param]))

    def set_prompt(
        self,
        prompt_name: Optional[str] = None,
        prompt_text: Optional[str] = None,
        **kwargs,
    ):
        if prompt_name is None and self.prompt is None:
            raise ValueError("Please set a value for the prompt")

        if self.prompt is None or self.prompt.name != prompt_name:
            self.prompt = Prompt(prompt_name, prompt_text)

        self.prompt.text = self.prompt.format(**kwargs)

    def run(self, **kwargs):
        if self.api_key is None:
            raise ValueError("Please provide an API key or set the API_KEY environment variable.")

        params = {**self.additional_params, **kwargs}
        client = OpenAI(api_key=self.api_key)

        if self.model_type == "openai":
            if self.model_name.startswith("whisper"):
                file = params.get("file")
                if file is None:
                    raise ValueError("Audio file data must be provided for transcription.")
                return client.audio.transcriptions.create(model=self.model_name, file=file)

            elif self.model_name in ["tts-1", "tts-1-hd"]:
                return client.audio.speech.create(
                    model=self.model_name,
                    voice=params.get("voice", self.voice),
                    input=params.get("input"),
                )

            if self.prompt is not None:
                empty_variables = self.prompt.get_variables()
                if empty_variables != []:
                    raise ValueError(f"Please provide values for the following variables: {empty_variables}")

                messages = [{"role": "system", "content": self.prompt.text}]
                return client.completions.create(
                    model=self.model_name,
                    messages=messages,
                    **params,
                )


def get_model(
    api_key: Optional[str] = None,
    model_type: Optional[str] = None,
    model_name: Optional[str] = None,
    **kwargs,
) -> LLM:
    return LLM(
        api_key=api_key or os.getenv("API_KEY"),
        model_type=model_type or os.getenv("MODEL_TYPE") or "openai",
        model_name=model_name or os.getenv("MODEL_NAME") or "gpt-3.5-turbo",
        **kwargs,
    )


def get_prompts_list() -> list:
    all_prompts = [
        pfile.stem
        for pfile in impresources.files(prompts).iterdir()
        if pfile.suffix == ".txt" and pfile.stem.endswith("_pt")
    ]
    return all_prompts