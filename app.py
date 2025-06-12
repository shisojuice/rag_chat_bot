import streamlit as st
from rag_chain import RAGChatBot
from utils.model_selector import get_available_models
import os
from dotenv import load_dotenv

load_dotenv()

# VectorDBの保存先ディレクトリとドキュメントの保存先ディレクトリを設定
os.environ["VECTOR_STORE_PATH"] = os.getenv("VECTOR_STORE_PATH", "vector")
os.environ["DOCUMENT_PATH"] = os.getenv("DOCUMENT_PATH","docs")
os.makedirs( os.environ["VECTOR_STORE_PATH"], exist_ok=True)
os.makedirs( os.environ["DOCUMENT_PATH"], exist_ok=True)

st.set_page_config(page_title="社内資料RAG ChatBot", layout="wide")
st.title("社内資料RAG ChatBot")

# sidebarに仕切りを作成する
st.sidebar.markdown("---")     
st.sidebar.header("AIモデル設定")

# モデル選択
models = get_available_models()
def_model = st.session_state.get("AI_MODEL", os.getenv("AI_MODEL", "openai"))
model_name = st.sidebar.selectbox("AIモデルを選択", models, index=models.index(def_model) if def_model in models else 0, key="ai_model_select")

# モデル設定用の入力欄
ai_model_inputs = {}
if model_name == "openai":
    ai_model_inputs["OPENAI_API_KEY"] = st.sidebar.text_input("OpenAI APIキー", type="password", value=st.session_state.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", "")), key="openai_api_key")
    ai_model_inputs["OPENAI_MODEL_NAME"] = st.sidebar.text_input("OpenAI モデル名", value=st.session_state.get("OPENAI_MODEL_NAME", os.getenv("OPENAI_MODEL_NAME", "gpt-3.5-turbo")), key="openai_model_name")
elif model_name == "azure":
    ai_model_inputs["AZURE_OPENAI_API_KEY"] = st.sidebar.text_input("Azure OpenAI APIキー", type="password", value=st.session_state.get("AZURE_OPENAI_API_KEY", os.getenv("AZURE_OPENAI_API_KEY", "")), key="azure_api_key")
    ai_model_inputs["AZURE_OPENAI_ENDPOINT"] = st.sidebar.text_input("Azure OpenAI エンドポイント", value=st.session_state.get("AZURE_OPENAI_ENDPOINT", os.getenv("AZURE_OPENAI_ENDPOINT", "")), key="azure_endpoint")
    ai_model_inputs["AZURE_OPENAI_DEPLOYMENT"] = st.sidebar.text_input("Azure OpenAI デプロイメント名", value=st.session_state.get("AZURE_OPENAI_DEPLOYMENT", os.getenv("AZURE_OPENAI_DEPLOYMENT", "")), key="azure_deployment")
    ai_model_inputs["AZURE_OPENAI_API_VERSION"] = st.sidebar.text_input("Azure OpenAI APIバージョン", value=st.session_state.get("AZURE_OPENAI_API_VERSION", os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15")), key="azure_api_version")
    ai_model_inputs["AZURE_OPENAI_MODEL_NAME"] = st.sidebar.text_input("Azure OpenAI モデル名", value=st.session_state.get("AZURE_OPENAI_MODEL_NAME", os.getenv("AZURE_OPENAI_MODEL_NAME", "gpt-35-turbo")), key="azure_model_name")
elif model_name == "gemini":
    ai_model_inputs["GEMINI_API_KEY"] = st.sidebar.text_input("Gemini APIキー", type="password", value=st.session_state.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", "")), key="gemini_api_key")
    ai_model_inputs["GEMINI_MODEL_NAME"] = st.sidebar.text_input("Gemini モデル名", value=st.session_state.get("GEMINI_MODEL_NAME", os.getenv("GEMINI_MODEL_NAME", "gemini-pro")), key="gemini_model_name")

if st.sidebar.button("AIモデル設定を適用"):
    st.session_state["AI_MODEL"] = model_name
    os.environ["AI_MODEL"] = model_name
    for k, v in ai_model_inputs.items():
        st.session_state[k] = v
        os.environ[k] = v
    st.sidebar.success("AIモデル設定を適用しました。")

# 必須項目の警告
if model_name == "openai":
    if not ai_model_inputs["OPENAI_API_KEY"]:
        st.sidebar.warning("OpenAI APIキーを入力してください。")
elif model_name == "azure":
    if not ai_model_inputs["AZURE_OPENAI_API_KEY"] or not ai_model_inputs["AZURE_OPENAI_ENDPOINT"] or not ai_model_inputs["AZURE_OPENAI_DEPLOYMENT"] or not ai_model_inputs["AZURE_OPENAI_API_VERSION"]:
        st.sidebar.warning("Azure OpenAI APIキー、エンドポイント、デプロイメント名、APIバージョンを入力してください。")
elif model_name == "gemini":
    if not ai_model_inputs["GEMINI_API_KEY"]:
        st.sidebar.warning("Gemini APIキーを入力してください。")

# sidebarに仕切りを作成する
st.sidebar.markdown("---")        
st.sidebar.header("埋め込みモデル設定")

# 埋め込みモデルプロバイダー選択（openai or huggingface）
def_embed_provider = st.session_state.get("EMBEDDING_MODEL_PROVIDER", os.getenv("EMBEDDING_MODEL_PROVIDER", "openai"))
embed_model_provider = st.sidebar.selectbox(
    "埋め込みモデルプロバイダーを選択", 
    ["openai", "huggingface"], 
    index=["openai", "huggingface"].index(def_embed_provider),
    key="embed_model_provider_select"
)

# 埋め込みモデル名
if embed_model_provider == "openai":
    default_embed_model = st.session_state.get("EMBEDDING_MODEL", os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002"))
else:
    default_embed_model = st.session_state.get("EMBEDDING_MODEL", os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
embed_model = st.sidebar.text_input("埋め込みモデル名", value=default_embed_model, key="embed_model_name")

# 埋め込みAPIキー
embed_api_key = ""
if embed_model_provider == "openai":
    embed_api_key = st.sidebar.text_input(
        "OpenAI 埋め込みAPIキー", 
        type="password", 
        value=st.session_state.get("OPENAI_EMBED_API_KEY", os.getenv("OPENAI_EMBED_API_KEY", "")),
        key="openai_embed_api_key"
    )
elif embed_model_provider == "huggingface":
    embed_api_key = st.sidebar.text_input(
        "Hugging Face 埋め込みAPIキー", 
        type="password", 
        value=st.session_state.get("HUGGINGFACE_EMBED_API_KEY", os.getenv("HUGGINGFACE_EMBED_API_KEY", "")),
        key="huggingface_embed_api_key"
    )

if st.sidebar.button("埋め込みモデル設定を適用"):
    st.session_state["EMBEDDING_MODEL_PROVIDER"] = embed_model_provider
    os.environ["EMBEDDING_MODEL_PROVIDER"] = embed_model_provider
    st.session_state["EMBEDDING_MODEL"] = embed_model
    os.environ["EMBEDDING_MODEL"] = embed_model
    if embed_model_provider == "openai":
        st.session_state["OPENAI_EMBED_API_KEY"] = embed_api_key
        os.environ["OPENAI_EMBED_API_KEY"] = embed_api_key
    elif embed_model_provider == "huggingface":
        st.session_state["HUGGINGFACE_EMBED_API_KEY"] = embed_api_key
        os.environ["HUGGINGFACE_EMBED_API_KEY"] = embed_api_key
    st.sidebar.success("埋め込みモデル設定を適用しました。")

# 必須項目の警告
if embed_model_provider == "openai":
    if not embed_api_key:
        st.sidebar.warning("OpenAI 埋め込みAPIキーを入力してください。")
elif embed_model_provider == "huggingface":
    if not embed_api_key:
        st.sidebar.warning("Hugging Face 埋め込みAPIキーを入力してください。")

# チャットボット初期化
bot = RAGChatBot(model_name)

# チャット履歴
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# 入力
user_input = st.text_input("質問を入力してください", key="input")
if st.button("送信") and user_input:
    with st.spinner("回答生成中..."):
        answer = bot.ask(user_input)
        st.session_state["messages"].append((user_input, answer))

# 履歴表示
for q, a in reversed(st.session_state["messages"]):
    st.markdown(f"**Q:** {q}")
    st.markdown(f"**A:** {a}")


# sidebarに仕切りを作成する
st.sidebar.markdown("---")       
# ファイルアップロード
st.sidebar.header("資料アップロード")
uploaded_files = st.sidebar.file_uploader("Txt/Md/Tsv/Csv/Excel/Pdfをアップロード", type=["pdf", "xlsx", "xls", "txt","md", "tsv","csv"], accept_multiple_files=True)
if st.sidebar.button("ベクトル化") and uploaded_files:
    from utils.loader import save_and_embed_files
    with st.spinner("ベクトル化中..."):
        save_and_embed_files(uploaded_files)
    st.sidebar.success("ベクトル化完了！")


