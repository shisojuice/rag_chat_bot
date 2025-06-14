import os
from dotenv import load_dotenv
from langchain_community.llms import OpenAI
from langchain.llms.base import LLM
import requests
from pydantic import BaseModel

# Gemini, AzureOpenAIのAPI呼び出しを実装

def get_available_models():
    return ["azure"]

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
        # URLのスラッシュ重複を防ぐ
        deployment = self.deployment or ""
        model_name = os.environ.get("AZURE_OPENAI_MODEL_NAME", "")
        endpoint = (self.endpoint or "").rstrip("/")
        api_version = self.api_version
        headers = {
            "Content-Type": "application/json",
            "api-key": self.api_key
        }
        # gpt系モデルはchat/completionsエンドポイント+messages形式
        if any(x in deployment.lower() for x in ["gpt", "turbo"]) or any(x in model_name.lower() for x in ["gpt", "turbo"]):
            url = f"{endpoint}/openai/deployments/{deployment}/chat/completions?api-version={api_version}"
            data = {
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 512,
                "temperature": 0
            }
        else:
            url = f"{endpoint}/openai/deployments/{deployment}/completions?api-version={api_version}"
            data = {
                "prompt": prompt,
                "max_tokens": 512,
                "temperature": 0
            }
        resp = requests.post(url, headers=headers, json=data)
        resp.raise_for_status()
        # chat形式はchoices[0]["message"]["content"]、completion形式はchoices[0]["text"]
        result = resp.json()
        if "choices" in result:
            if "message" in result["choices"][0]:
                return result["choices"][0]["message"]["content"]
            elif "text" in result["choices"][0]:
                return result["choices"][0]["text"]
        return str(result)


def get_llm(model_name):
    load_dotenv()
    if model_name == "azure":
        if not (os.environ.get("AZURE_OPENAI_API_KEY") and os.environ.get("AZURE_OPENAI_ENDPOINT") and os.environ.get("AZURE_OPENAI_DEPLOYMENT")):
            raise ValueError("AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENTが設定されていません")
        api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2023-05-15")
        return AzureOpenAILLM(api_version=api_version)
    else:
        raise ValueError("未知のモデルです")
