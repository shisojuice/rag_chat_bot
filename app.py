import streamlit as st
from rag_chain import RAGChatBot
from utils.model_selector import get_available_models
import os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="社内資料RAG ChatBot", layout="wide")
st.title("社内資料RAG ChatBot")

# モデル選択
models = get_available_models()
def_model = os.getenv("AI_MODEL", "openai")
model_name = st.sidebar.selectbox("AIモデルを選択", models, index=models.index(def_model) if def_model in models else 0)

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

# ファイルアップロード
st.sidebar.header("資料アップロード")
uploaded_files = st.sidebar.file_uploader("Excel/PDF/txtをアップロード", type=["pdf", "xlsx", "xls", "txt"], accept_multiple_files=True)
if st.sidebar.button("ベクトル化") and uploaded_files:
    from utils.loader import save_and_embed_files
    with st.spinner("ベクトル化中..."):
        save_and_embed_files(uploaded_files)
    st.sidebar.success("ベクトル化完了！")
