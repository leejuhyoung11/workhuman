from openai import OpenAI

from src.models.base_wrapper import BaseLLMWrapper

class OpenAIWrapper(BaseLLMWrapper):

    def new_client(self):
        return OpenAI(api_key=self.api_key)

    def _invoke(self, client, prompt: str) -> str:
        response = client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message["content"]