from langchain_huggingface import HuggingFaceEmbeddings

from src import config


def build_embedding_model() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
