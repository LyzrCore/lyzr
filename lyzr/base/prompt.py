from lyzr.base import prompts
from typing import Optional, Union
from importlib import resources as impresources
from lyzr.base.errors import MissingValueError, InvalidValueError


class Prompt:
    def __init__(self, prompt_name: str, prompt_text: Optional[str] = None):
        self.name = prompt_name
        self.text = prompt_text
        if self.text is None:
            # in-built prompt names end with _pt
            self.load_prompt()
            self.variables = self.get_variables()
            return None
        else:
            self.variables = self.get_variables()
            self.save_prompt()
            return None

    def get_variables(self):
        variables = []
        for word in self.text.split():
            if word.startswith("{") and word.endswith("}"):
                variables.append(word[1:-1])
        return variables

    def save_prompt(self):
        inp_file = impresources.files(prompts) / f"{self.name}.txt"
        with inp_file.open("wb") as f:
            f.write(self.text.encode("utf-8"))

    def load_prompt(self):
        try:
            inp_file = impresources.files(prompts) / f"{self.name}.txt"
            with inp_file.open("rb") as f:
                self.text = f.read().decode("utf-8")
        except FileNotFoundError:
            raise ValueError(
                f"No prompt with name '{self.name}' found. "
                f"To use an in-built prompt, use one of the following prompt names: {get_prompts_list()}\n"
                "Or create a new prompt by passing the prompt text.",
            )

    def edit_prompt(self, prompt_text: str):
        self.text = prompt_text
        self.variables = self.get_variables()
        self.save_prompt()
        return self

    def format(self, **kwargs):
        if self.text is None:
            raise ValueError(f"Please provide the text for the prompt '{self.name}'")
        prompt_text = self.text
        try:
            prompt_text = prompt_text.format(**kwargs)
        except KeyError:
            print(f"Please provide values for all variables: {self.variables}")
            raise
        self.text = prompt_text
        return self


def get_prompts_list() -> list:
    # fix this path issue
    all_prompts = [
        pfile.stem
        for pfile in impresources.files(prompts).iterdir()
        if (pfile.suffix == ".txt") and (pfile.stem.endswith("_pt"))
    ]
    return all_prompts


def get_prompt_text(prompt: Union[dict, Prompt]):
    if isinstance(prompt, Prompt):
        return prompt.text
    if not isinstance(prompt, dict):
        raise InvalidValueError(["dict", "Prompt"])
    if ("prompt" not in prompt) and ("text" not in prompt):
        raise MissingValueError(["prompt", "text"])
    if "prompt" in prompt:
        return get_prompt_text(prompt["prompt"])
    elif "name" in prompt:
        return Prompt(prompt["name"], prompt["text"]).text
    else:
        return prompt["text"]
