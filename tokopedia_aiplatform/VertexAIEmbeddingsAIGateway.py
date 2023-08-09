from typing import Dict
from langchain.embeddings.base import Embeddings
from langchain.llms.vertexai import _VertexAICommon
from pydantic import root_validator


class VertexAIEmbeddingsAIGateway(_VertexAICommon, Embeddings):
    """Google Cloud VertexAI embedding models."""

    model_name: str = "textembedding-gecko"

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        return values