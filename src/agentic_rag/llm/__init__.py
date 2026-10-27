from .base import LLMClient, LLMError, LLMResponse, Message, ToolCall, ToolSpec, Usage, complete_json, extract_json

__all__ = ["LLMClient", "LLMError", "LLMResponse", "Message", "ToolCall", "ToolSpec", "Usage", "complete_json", "extract_json", "make_llm"]


def make_llm(settings, *, fast: bool = False):
    """Tạo client theo cấu hình. fast=True dùng model rẻ cho router/critic."""
    model = settings.fast_model if fast else settings.main_model
    if settings.llm_provider == "gemini":
        from .gemini_client import GeminiClient
        return GeminiClient(default_model=model)
    if settings.llm_provider == "anthropic":
        from .anthropic_client import AnthropicClient
        return AnthropicClient(default_model=model)
    if settings.llm_provider == "mock":
        from .mock import MockLLM
        return MockLLM()
    raise ValueError(f"LLM_PROVIDER không hợp lệ: {settings.llm_provider}")
