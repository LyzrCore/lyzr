class ImproperUsageError(ValueError):
    """Raise for improper usage or missing params."""

    pass


class MissingValueError(ValueError):
    """Raise for missing values."""

    pass


class InvalidModelError(ValueError):
    """Raise for invalid model"""

    pass


class DependencyError(ImportError):
    """Raise for missing modules"""

    def __init__(self, required_modules: dict):
        super().__init__(
            f"The following modules are needed to run this function: {', '.join(required_modules.keys())}. Please install them using: `pip install {' '.join(required_modules.values())}`"
        )


class ImproperlyConfigured(Exception):
    """Raise for incorrect configuration."""

    pass


class ValidationError(Exception):
    """Raise for validations"""

    pass
