from __future__ import annotations

import hashlib
from typing import List


class Embedder:
    def __init__(self, model_name: str, device: str = "cuda", dim: int = 384) -> None:
        self.model_name = model_name
        self.dim = dim
        self.model = None
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name, device=device)
            probe = self.embed(["probe"])[0]
            self.dim = len(probe)
        except Exception as exc:
            print(f"[WARN] SentenceTransformer unavailable, using hash embeddings: {exc}")

    def _hash_embedding(self, text: str) -> List[float]:
        values = [0.0] * self.dim
        for token in text.lower().split():
            digest = hashlib.md5(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "little") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            values[idx] += sign
        norm = sum(v * v for v in values) ** 0.5 or 1.0
        return [v / norm for v in values]

    def embed(self, texts: List[str]) -> List[List[float]]:
        if self.model is None:
            return [self._hash_embedding(t) for t in texts]
        emb = self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return emb.astype("float32").tolist()

