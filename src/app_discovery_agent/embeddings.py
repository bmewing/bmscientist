from __future__ import annotations

from sentence_transformers import SentenceTransformer

from app_discovery_agent.config import AppConfig


class LocalEmbedder:
    def __init__(self, config: AppConfig):
        self._model = SentenceTransformer(config.embedding_model)

    @property
    def dimension(self) -> int:
        return int(self._model.get_sentence_embedding_dimension())

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._model.encode(texts, normalize_embeddings=True)
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

