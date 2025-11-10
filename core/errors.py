"""Custom error classes for the deep research agent."""


class DeepResearchError(Exception):
    """Base error class for deep research agent."""
    
    def __init__(self, message: str, status_code: int = 500):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)


class APIKeyError(DeepResearchError):
    """Raised when API key is missing or invalid."""
    
    def __init__(self, message: str):
        super().__init__(message, status_code=401)


class LLMServiceError(DeepResearchError):
    """Raised when LLM service call fails."""
    pass


class SearchServiceError(DeepResearchError):
    """Raised when search service call fails."""
    pass

