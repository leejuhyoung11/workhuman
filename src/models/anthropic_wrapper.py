from anthropic import Anthropic

from src.models.base_wrapper import BaseLLMWrapper

class AnthropicWrapper(BaseLLMWrapper):

    def new_client(self):
        return Anthropic(api_key=self.api_key)

    def _invoke(self, client, prompt: str) -> str:
        response = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text