import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
    
def get_embedding_model():
    provider = os.environ.get("EMBEDDING_MODEL_PROVIDER", "azure").lower()
    if provider == "azure":
        from langchain_openai import AzureOpenAIEmbeddings
        return AzureOpenAIEmbeddings(
            azure_endpoint=os.environ.get("AZURE_OPENAI_EMBED_ENDPOINT"),
            azure_deployment=os.environ.get("AZURE_OPENAI_EMBED_DEPLOYMENT"),
            api_key=os.environ.get("AZURE_OPENAI_EMBED_API_KEY"),
            api_version=os.environ.get("AZURE_OPENAI_EMBED_API_VERSION", "2024-02-01"),
        )
    else:
        raise ValueError(f"未対応の埋め込みモデルプロバイダーです: {provider}")
