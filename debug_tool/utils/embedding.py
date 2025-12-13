import logging
from chromadb import Documents, EmbeddingFunction, Embeddings
from openai import OpenAI

logger = logging.getLogger(__name__)

class OpenAIEmbeddingFunction(EmbeddingFunction):
    def __init__(self, api_key: str, base_url: str, model_name: str, dimensions: int = None):
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name
        self.dimensions = dimensions

        # Configure OpenAI Client
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

    def __call__(self, input: Documents) -> Embeddings:
        """
        Generate embeddings for a list of documents.
        """
        # Ensure input is a list
        if isinstance(input, str):
            input = [input]

        # Filter out empty strings to avoid API errors, but keep indices aligned?
        # Chroma expects 1:1 mapping. If input is empty, return zero vector?
        # OpenAI usually handles empty string with error or weird result.
        # Let's assume input is valid text.

        try:
            # Prepare arguments
            kwargs = {
                "input": input,
                "model": self.model_name
            }
            if self.dimensions:
                kwargs["dimensions"] = self.dimensions

            # Call API
            response = self.client.embeddings.create(**kwargs)

            # Extract embeddings
            # response.data is a list of objects, we need to sort by index just in case, though usually ordered
            sorted_data = sorted(response.data, key=lambda x: x.index)
            return [item.embedding for item in sorted_data]

        except Exception as e:
            logger.error(f"Error calling Embedding API: {e}")
            raise e

def create_embedding_function(provider_config: dict) -> EmbeddingFunction:
    """
    Factory function to create the appropriate embedding function from config.
    """
    if not provider_config:
        raise ValueError("Provider configuration is empty")

    # Extract required fields
    # Handle 'key' which might be a list or string
    api_key = provider_config.get("key")
    if isinstance(api_key, list):
        api_key = api_key[0]
    elif not api_key:
        # Fallback for some providers like SiliconFlow/OpenAI-compatible where key might be in embedding_api_key
        api_key = provider_config.get("embedding_api_key")

    api_base = provider_config.get("api_base") or provider_config.get("embedding_api_base")
    model = provider_config.get("model_config", {}).get("model") or provider_config.get("embedding_model")
    dimensions = provider_config.get("embedding_dimensions") # Optional

    if not api_key or not api_base or not model:
        raise ValueError(f"Incomplete configuration for provider: {provider_config.get('id')}. Need key, api_base, and model.")

    return OpenAIEmbeddingFunction(
        api_key=api_key,
        base_url=api_base,
        model_name=model,
        dimensions=dimensions
    )