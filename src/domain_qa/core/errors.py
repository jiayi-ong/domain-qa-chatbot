class AgentError(Exception):
    """Base exception for predictable agent failures."""


class ConfigError(AgentError):
    pass


class LLMError(AgentError):
    pass


class PostCheckError(AgentError):
    pass