from typing import Union


class MissingValueError(ValueError):
    def __init__(self, params: Union[str, list]):
        super().__init__(f"Required value is missing. Provide one of: {params}")


class InvalidModelError(ValueError):
    def __init__(self):
        super().__init__("Invalid model provided.")


class InvalidValueError(ValueError):
    def __init__(self, params: list):
        super().__init__(f"Invalid value provided. Provide value of type: {params}")


class MissingModuleError(ImportError):
    def __init__(self, required_modules: dict):
        super().__init__(
            f"The following modules are needed to run this function: {', '.join(required_modules.keys())}. Please install them using: `pip install {' '.join(required_modules.values())}`"
        )
