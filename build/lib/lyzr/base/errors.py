from typing import Union

from lyzr.base.llms import LLM


class MissingValueError(ValueError):
    def __init__(self, params: Union[str, list]):
        super().__init__(f"Required value is missing. Provide one of: {params}")


class InvalidModelError(ValueError):
    def __init__(self):
        super().__init__("Invalid model provided.")


class InvalidValueError(ValueError):
    def __init__(self, params: list):
        super().__init__(f"Invalid value provided. Provide value of type: {params}")


def check_values(
    query: Union[str, None],
    model: Union[LLM, None],
    params: dict,
) -> None:
    if query is None:
        raise MissingValueError(["query"])

    if model is not None:
        return None

    for value in params.values():
        if value is None:
            raise MissingValueError(["model or ", ", ".join(params.keys())])
