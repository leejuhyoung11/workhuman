class BaseLLMWrapper:
    def __init__(self, model, api_key, temperature=0, max_tokens=4000):
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens

    def new_client(self):
        """Return a new API client every call."""
        raise NotImplementedError

    def call(self, prompt: str) -> str:
        """Return raw text output from LLM."""
        client = self.new_client()
        return self._invoke(client, prompt)

    def _invoke(self, client, prompt: str) -> str:
        raise NotImplementedError