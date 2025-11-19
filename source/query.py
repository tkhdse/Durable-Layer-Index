from sentence_transformers import SentenceTransformer
import asyncio
from sentence_transformers import SentenceTransformer

EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

class Query:
    def __init__(self, text: str, future: asyncio.Future):
        self.text = text
        self.vector_embedding = EMBED_MODEL.encode(text, normalize_embeddings=True)
        self.future = future

    def get_text(self):
        return self.text

    def get_embedding(self):
        return self.vector_embedding

    def get_future(self):
        return self.future