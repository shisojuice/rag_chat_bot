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
# st.markdown(
#     """
#     <link rel="shortcut icon" href="favicon.ico">
#     """,
#     unsafe_allow_html=True
# )

# ページ切り替え
page = st.sidebar.radio("ページ選択", ("チャット", "資料管理", "AIモデル設定", "埋め込みモデル設定"), key="page_select")
st.title("社内資料RAG ChatBot")

def ai_model_settings():
    st.header("AIモデル設定")
    models = get_available_models()
    def_model = st.session_state.get("AI_MODEL", os.getenv("AI_MODEL", "azure"))
    model_name = st.selectbox("AIモデルを選択", models, index=models.index(def_model) if def_model in models else 0, key="ai_model_select_page")
    ai_model_inputs = {}
    if model_name == "azure":
        ai_model_inputs["AZURE_OPENAI_API_KEY"] = st.text_input("Azure OpenAI APIキー", type="password", value=st.session_state.get("AZURE_OPENAI_API_KEY", os.getenv("AZURE_OPENAI_API_KEY", "")), key="azure_api_key_page")
        ai_model_inputs["AZURE_OPENAI_ENDPOINT"] = st.text_input("Azure OpenAI エンドポイント", value=st.session_state.get("AZURE_OPENAI_ENDPOINT", os.getenv("AZURE_OPENAI_ENDPOINT", "")), key="azure_endpoint_page")
        ai_model_inputs["AZURE_OPENAI_MODEL_NAME"] = st.text_input("Azure OpenAI モデル名", value=st.session_state.get("AZURE_OPENAI_MODEL_NAME", os.getenv("AZURE_OPENAI_MODEL_NAME", "gpt-35-turbo")), key="azure_model_name_page")
        ai_model_inputs["AZURE_OPENAI_DEPLOYMENT"] = st.text_input("Azure OpenAI デプロイメント名", value=st.session_state.get("AZURE_OPENAI_DEPLOYMENT", os.getenv("AZURE_OPENAI_DEPLOYMENT", "")), key="azure_deployment_page")
        ai_model_inputs["AZURE_OPENAI_API_VERSION"] = st.text_input("Azure OpenAI APIバージョン", value=st.session_state.get("AZURE_OPENAI_API_VERSION", os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15")), key="azure_api_version_page")
    if st.button("AIモデル設定を適用", key="ai_model_apply_page"):
        st.session_state["AI_MODEL"] = model_name
        os.environ["AI_MODEL"] = model_name
        for k, v in ai_model_inputs.items():
            st.session_state[k] = v
            os.environ[k] = v
        st.success("AIモデル設定を適用しました。")
def embedding_model_settings():
    st.header("埋め込みモデル設定")
    def_embed_provider = st.session_state.get("EMBEDDING_MODEL_PROVIDER", os.getenv("EMBEDDING_MODEL_PROVIDER", "azure"))
    # 選択肢にない場合は強制的に"azure"にする
    if def_embed_provider not in ["azure"]:
        def_embed_provider = "azure"
    embed_model_provider = st.selectbox(
        "埋め込みモデルプロバイダーを選択", 
        ["azure"], 
        index=["azure"].index(def_embed_provider),
        key="embed_model_provider_select_page"
    )
    if embed_model_provider == "azure":
        default_embed_model = st.session_state.get("EMBEDDING_MODEL", os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"))
    else:
        default_embed_model = st.session_state.get("EMBEDDING_MODEL", os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
    embed_api_key = ""
    if embed_model_provider == "azure":
        embed_api_key = st.text_input("Azure OpenAI 埋め込みAPIキー", type="password", value=st.session_state.get("AZURE_OPENAI_EMBED_API_KEY", os.getenv("AZURE_OPENAI_EMBED_API_KEY", "")),key="azure_embed_api_key_page")
        embed_endpoint = st.text_input("Azure OpenAI 埋め込みエンドポイント", value=st.session_state.get("AZURE_OPENAI_EMBED_ENDPOINT", os.getenv("AZURE_OPENAI_EMBED_ENDPOINT", "")),key="azure_embed_endpoint_page")
        embed_model_name = st.text_input("Azure OpenAI 埋め込みモデル名", value=st.session_state.get("AZURE_OPENAI_EMBED_MODEL_NAME", os.getenv("AZURE_OPENAI_EMBED_MODEL_NAME", "text-embedding-3-small")),key="azure_embed_model_name_page")
        embed_deployment = st.text_input("Azure OpenAI 埋め込みデプロイメント名", value=st.session_state.get("AZURE_OPENAI_EMBED_DEPLOYMENT", os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT", "text-embedding-3-small")),key="azure_embed_deployment_page")
        embed_api_version = st.text_input("Azure OpenAI 埋め込みAPIバージョン", value=st.session_state.get("AZURE_OPENAI_EMBED_API_VERSION", os.getenv("AZURE_OPENAI_EMBED_API_VERSION", "2024-02-01")),key="azure_embed_api_version_page")
    if st.button("埋め込みモデル設定を適用", key="embed_model_apply_page"):
        st.session_state["EMBEDDING_MODEL_PROVIDER"] = embed_model_provider
        os.environ["EMBEDDING_MODEL_PROVIDER"] = embed_model_provider
        for k, v in embed_model_inputs.items():
            st.session_state[k] = v
            os.environ[k] = v
        st.success("埋め込みモデル設定を適用しました。")
    

# ページごとにUIを分岐
if page == "AIモデル設定":
    ai_model_settings()
elif page == "埋め込みモデル設定":
    embedding_model_settings()
else:
    # AIモデル名をセッションまたは環境変数から取得
    model_name = st.session_state.get("AI_MODEL", os.getenv("AI_MODEL", "azure"))
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

    elif page == "チャット":
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
