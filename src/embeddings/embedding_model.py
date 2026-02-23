from langchain_huggingface import HuggingFaceEmbeddings

from config import settings


class EmbeddingModel:
    def __init__(self, model_name=None, device=None):
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self.device = device or settings.EMBEDDING_DEVICE
        self._embeddings = None

    def get_embeddings(self):
        if self._embeddings is None:
            self._embeddings = HuggingFaceEmbeddings(
                model_name=self.model_name,
                model_kwargs={"device": self.device},
                encode_kwargs={"normalize_embeddings": True},
            )
        return self._embeddings

    def embed_query(self, text):
        return self.get_embeddings().embed_query(text)

    def embed_documents(self, texts):
        return self.get_embeddings().embed_documents(texts)


if __name__ == "__main__":
    model = EmbeddingModel()
    print(f"Model: {model.model_name}")
    print(f"Device: {model.device}")
