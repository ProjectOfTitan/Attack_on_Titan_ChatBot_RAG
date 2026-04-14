from __future__ import annotations

from typing import Iterable

from langchain_core.embeddings import Embeddings


class LocalHFEmbeddings(Embeddings):
    def __init__(
        self,
        model_name: str,
        *,
        device: str = "cpu",
        normalize: bool = True,
        query_prefix: str = "",
        document_prefix: str = "",
        max_length: int = 512,
        trust_remote_code: bool = False,
    ) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._torch = torch
        self.model_name = model_name
        self.device = device
        self.normalize = normalize
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
        )
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
        )
        self.model.to(device)
        self.model.eval()

    def _mean_pool(
        self,
        last_hidden_state,
        attention_mask,
    ):
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        masked = last_hidden_state * mask
        summed = masked.sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    def _embed(self, texts: Iterable[str], *, prefix: str) -> list[list[float]]:
        prefixed_texts = [f"{prefix}{text}".strip() for text in texts]
        encoded = self.tokenizer(
            prefixed_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}

        with self._torch.no_grad():
            outputs = self.model(**encoded)

        embeddings = self._mean_pool(outputs.last_hidden_state, encoded["attention_mask"])
        if self.normalize:
            embeddings = self._torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings.cpu().tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts, prefix=self.document_prefix)

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], prefix=self.query_prefix)[0]
