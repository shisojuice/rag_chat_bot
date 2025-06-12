import streamlit as st
from rag_chain import RAGChatBot
from utils.model_selector import get_available_models
import os
from dotenv import load_dotenv
import shutil

load_dotenv()

# VectorDBの保存先ディレクトリとドキュメントの保存先ディレクトリを設定
os.environ["VECTOR_STORE_PATH"] = os.getenv("VECTOR_STORE_PATH", "vector")
os.environ["DOCUMENT_PATH"] = os.getenv("DOCUMENT_PATH","docs")
os.makedirs( os.environ["VECTOR_STORE_PATH"], exist_ok=True)
os.makedirs( os.environ["DOCUMENT_PATH"], exist_ok=True)

st.set_page_config(page_title="社内資料RAG ChatBot", layout="wide")

# --- NeoBrutalism + DarkMode CSS適用 ---
with open("neobrutalism.css", encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ページ切り替え
page = st.sidebar.radio("ページ選択", ("チャット", "資料管理"), key="page_select")
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

if page == "資料管理":
    st.header("資料アップロード＆管理")
    # アップロードUI
    st.subheader("資料アップロード")
    uploaded_files = st.file_uploader("Txt/Md/Tsv/Csv/Excel/Pdfをアップロード", type=["pdf", "xlsx", "xls", "txt","md", "tsv","csv"], accept_multiple_files=True)
    if st.button("ベクトル化") and uploaded_files:
        from utils.loader import save_and_embed_files
        with st.spinner("ベクトル化中..."):
            save_and_embed_files(uploaded_files)
        st.success("ベクトル化完了！")
    st.markdown("---")
    st.subheader("登録済み資料一覧")
    doc_list = bot.list_documents()
    if not doc_list:
        st.info("ベクトルストアに資料が登録されていません。")
    else:
        import pandas as pd
        doc_df = pd.DataFrame(doc_list)
        doc_df = doc_df.drop_duplicates(subset=["source_file", "folder_path", "uploaded_at"])  # 重複排除
        for i, row in doc_df.iterrows():
            col1, col2, col3, col4 = st.columns([4, 4, 4, 2])
            with col1:
                st.markdown(f"**ファイル名:** `{row['source_file']}`")
            with col2:
                st.markdown(f"**フォルダパス:** `{row['folder_path']}`")
            with col3:
                st.markdown(f"**アップロード日時:** `{row['uploaded_at']}`")
            with col4:
                if st.button("削除", key=f"delete_{row['source_file']}_{row['folder_path']}_{row['uploaded_at']}"):
                    # ファイル削除処理
                    file_path = os.path.join(os.environ["DOCUMENT_PATH"], row['source_file'])
                    if os.path.exists(file_path):
                        os.remove(file_path)
                    st.success(f"{row['source_file']} を削除しました。ベクトルストアを再構築してください。")
        st.warning("資料を削除した場合、再度ベクトル化ボタンを押してベクトルストアを再構築してください。")

else:
    # sidebarに仕切りを作成する
    st.sidebar.markdown("---")       

    import json
    CHAT_HISTORY_FILE = "chat_history.json"

    # 履歴の自動ロード
    if "messages" not in st.session_state:
        if os.path.exists(CHAT_HISTORY_FILE):
            try:
                with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as f:
                    st.session_state["messages"] = json.load(f)
            except Exception:
                st.session_state["messages"] = []
        else:
            st.session_state["messages"] = []

    # 履歴全削除ボタン
    if st.button("チャット履歴を全削除", key="clear_chat_history"):
        st.session_state["messages"] = []
        if os.path.exists(CHAT_HISTORY_FILE):
            os.remove(CHAT_HISTORY_FILE)
        st.success("チャット履歴を削除しました")

    # 入力
    user_input = st.text_input("質問を入力してください", key="input")
    if st.button("送信") and user_input:
        with st.spinner("回答生成中..."):
            answer = bot.ask(user_input)
            st.session_state["messages"].append((user_input, answer))
        # 保存
        try:
            with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
                json.dump(st.session_state["messages"], f, ensure_ascii=False, indent=2)
        except Exception as e:
            st.warning(f"履歴保存エラー: {e}")

    # 履歴表示
    for q, a in reversed(st.session_state["messages"]):
        st.markdown(f"**Q:** {q}")
        st.markdown(f"**A:** {a}")
