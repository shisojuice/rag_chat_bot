import os
from dotenv import load_dotenv
from langchain_community.llms import OpenAI
from langchain.llms.base import LLM
import requests
from pydantic import BaseModel

# Gemini, AzureOpenAIのAPI呼び出しを実装

def get_available_models():
    return ["openai", "gemini", "azure"]

class GeminiLLM(BaseModel):
    model: str

    @property
    def api_key(self):
        return os.environ.get("GEMINI_API_KEY")

    @property
    def _llm_type(self):
        return "gemini"

    def _call(self, prompt, stop=None, run_manager=None):
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        headers = {"Content-Type": "application/json"}
        data = {"contents": [{"parts": [{"text": prompt}]}]}
        resp = requests.post(url, headers=headers, json=data)
        resp.raise_for_status()
        return resp.json()["candidates"][0]["content"]["parts"][0]["text"]

class AzureOpenAILLM(BaseModel):
    api_version: str = "2023-05-15"

    @property
    def api_key(self):
        return os.environ.get("AZURE_OPENAI_API_KEY")

    @property
    def endpoint(self):
        return os.environ.get("AZURE_OPENAI_ENDPOINT")

    @property
    def deployment(self):
        return os.environ.get("AZURE_OPENAI_DEPLOYMENT")

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
        return OpenAI(openai_api_key=os.environ.get("OPENAI_API_KEY"), temperature=0)
    elif model_name == "gemini":
        model = os.environ.get("GEMINI_MODEL", "gemini-pro")
        if not os.environ.get("GEMINI_API_KEY"):
            raise ValueError("GEMINI_API_KEYが設定されていません")
        return GeminiLLM(model=model)
    elif model_name == "azure":
        if not (os.environ.get("AZURE_OPENAI_API_KEY") and os.environ.get("AZURE_OPENAI_ENDPOINT") and os.environ.get("AZURE_OPENAI_DEPLOYMENT")):
            raise ValueError("AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENTが設定されていません")
        api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2023-05-15")
        return AzureOpenAILLM(api_version=api_version)
    else:
        raise ValueError("未知のモデルです")
