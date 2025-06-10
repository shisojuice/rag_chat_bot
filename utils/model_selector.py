import os
from dotenv import load_dotenv
from langchain_community.llms import OpenAI
from langchain.llms.base import LLM
import requests

# Gemini, AzureOpenAIのAPI呼び出しを実装

def get_available_models():
    return ["openai", "gemini", "azure"]

class GeminiLLM(LLM):
    def __init__(self, api_key: str, model: str = "gemini-pro"):  # model名は必要に応じて変更
        self.api_key = api_key
        self.model = model
    @property
    def _llm_type(self):
        return "gemini"
    def _call(self, prompt, stop=None, run_manager=None):
        # Gemini API呼び出し例（Google Generative Language API）
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        headers = {"Content-Type": "application/json"}
        data = {"contents": [{"parts": [{"text": prompt}]}]}
        resp = requests.post(url, headers=headers, json=data)
        resp.raise_for_status()
        return resp.json()["candidates"][0]["content"]["parts"][0]["text"]

class AzureOpenAILLM(LLM):
    def __init__(self, api_key: str, endpoint: str, deployment: str, api_version: str = "2023-05-15"):
        self.api_key = api_key
        self.endpoint = endpoint
        self.deployment = deployment
        self.api_version = api_version
    @property
    def _llm_type(self):
        return "azureopenai"
    def _call(self, prompt, stop=None, run_manager=None):
        url = f"{self.endpoint}/openai/deployments/{self.deployment}/completions?api-version={self.api_version}"
        headers = {
            "Content-Type": "application/json",
            "api-key": self.api_key
        }
        data = {
            "prompt": prompt,
            "max_tokens": 512,
            "temperature": 0
        }
        resp = requests.post(url, headers=headers, json=data)
        resp.raise_for_status()
        return resp.json()["choices"][0]["text"]

def get_llm(model_name):
    load_dotenv()
    if model_name == "openai":
        return OpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"), temperature=0)
    elif model_name == "gemini":
        api_key = os.getenv("GEMINI_API_KEY")
        model = os.getenv("GEMINI_MODEL", "gemini-pro")
        if not api_key:
            raise ValueError("GEMINI_API_KEYが設定されていません")
        return GeminiLLM(api_key, model)
    elif model_name == "azure":
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15")
        if not (api_key and endpoint and deployment):
            raise ValueError("AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENTが設定されていません")
        return AzureOpenAILLM(api_key, endpoint, deployment, api_version)
    else:
        raise ValueError("未知のモデルです")
