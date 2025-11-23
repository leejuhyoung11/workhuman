from src.models.base_wrapper import BaseLLMWrapper
from src.models.anthropic_wrapper import AnthropicWrapper
from src.models.openai_wrapper import OpenAIWrapper
from src.models.gemini_wrapper import GoogleGeminiWrapper


class LLMProviderFactory:
    @staticmethod
    def create(provider: str, model: str, api_key: str, temperature: float = 0, max_tokens: int = 4000) -> BaseLLMWrapper:
        provider = provider.lower().strip()

        if provider == "anthropic":
            return AnthropicWrapper(
                model=model,
                api_key=api_key,
                temperature=temperature,
                max_tokens=max_tokens
            )

        elif provider == "openai":
            return OpenAIWrapper(
                model=model,
                api_key=api_key,
                temperature=temperature,
                max_tokens=max_tokens
            )

        elif provider == "gemini" or provider == "google":
            return GoogleGeminiWrapper(
                model=model,
                api_key=api_key,
                temperature=temperature,
                max_tokens=max_tokens
            )

        else:
            raise ValueError(f"[ERROR] Unknown LLM provider: {provider}")