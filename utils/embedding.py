import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

# HuggingFace埋め込み用
try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    huggingface_available = True
except ImportError:
    huggingface_available = False

def get_embedding_model():
    load_dotenv()
    provider = os.getenv("EMBEDDING_MODEL_PROVIDER", "openai").lower()
    if provider == "openai":
        return OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"), model=os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002"))
    elif provider == "huggingface":
        if not huggingface_available:
            raise ImportError("HuggingFaceEmbeddingsを利用するにはlangchain[hub]やsentence-transformersが必要です")
        model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        return HuggingFaceEmbeddings(model_name=model_name)
    else:
        raise ValueError(f"未対応の埋め込みモデルプロバイダーです: {provider}")
