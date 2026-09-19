import json
import os
from sentence_transformers import SentenceTransformer


class MigiEmbedder:
    """
    Embedding model wrapper for the DR. MIGI RAG pipeline.

    What it does:
        Loads BAAI/bge-small-en-v1.5 and exposes methods to convert text
        into 384-dimensional dense vectors (embeddings). These vectors
        capture semantic meaning — similar clinical text produces similar vectors,
        enabling semantic similarity search in the vector database.

    Why BAAI/bge-small-en-v1.5:
        - Runs fully locally — no API calls, no data leaving the machine
        - 130MB model — small enough for CPU inference
        - High retrieval quality (BEIR benchmark top performer at its size)
        - Apache 2.0 licence — free for any use
        - 384-dimensional output — good balance of precision and speed

    Why it is needed:
        Both the patient record chunks (at ingestion time) and the doctor's
        query (at inference time) must be converted to the same vector space
        using the same model, so that ChromaDB can compute meaningful
        similarity scores between them.

    Best practices:
        Always use the same embedding model for both ingestion and retrieval.
        Mixing models produces incomparable vector spaces and breaks retrieval.
    """

    def __init__(self, config_path: str = "configs/rag_config.json"):
        """
        Loads the embedding model specified in the RAG config.

        Arguments:
            config_path: Path to rag_config.json.
        """
        with open(config_path, "r") as f:
            config = json.load(f)

        model_name = config.get("embedding_model", "BAAI/bge-small-en-v1.5")

        # Redirect HuggingFace cache to project models/cache directory
        default_cache = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models", "cache")
        os.environ.setdefault("HF_HOME", default_cache)

        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        print(f"Embedding model loaded. Output dimensions: {self.model.get_sentence_embedding_dimension()}")

    def embed_text(self, text: str) -> list[float]:
        """
        Converts a single string into a vector embedding.

        Arguments:
            text: The input string to embed.

        Returns:
            A list of floats representing the 384-dimensional embedding vector.
        """
        # normalize_embeddings=True is recommended for bge models
        # It makes cosine similarity and dot product equivalent, improving retrieval
        embedding = self.model.encode(text, normalize_embeddings=True)
        return embedding.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Converts a list of strings into a list of embedding vectors.

        Why it is needed:
            Batch embedding is significantly faster than embedding one text at a time
            because the model processes multiple inputs in a single forward pass.
            Used during ingestion to embed all chunks efficiently.

        Arguments:
            texts: List of input strings.

        Returns:
            List of 384-dimensional embedding vectors, one per input string.
        """
        embeddings = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
        return embeddings.tolist()

    @property
    def embedding_dimension(self) -> int:
        """Returns the dimensionality of the embedding vectors produced by this model."""
        return self.model.get_sentence_embedding_dimension()
