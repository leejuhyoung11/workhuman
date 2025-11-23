import google.generativeai as genai

from src.models.base_wrapper import BaseLLMWrapper

class GoogleGeminiWrapper(BaseLLMWrapper):

    def new_client(self):
        genai.configure(api_key=self.api_key)
        return genai.GenerativeModel(self.model)

    def _invoke(self, model, prompt: str) -> str:
        response = model.generate_content(
            prompt,
            generation_config={"temperature": self.temperature}
        )
        return response.text