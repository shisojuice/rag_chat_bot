# 社内資料RAG ChatBot

## 概要
- Streamlit, LangChain, FAISSを使用した社内資料検索RAGチャットボット
- OpenAI, Gemini, AzureOpenAIのAIモデル選択対応
- Excel, PDF, txtをベクトル化しvector/配下に保存

## セットアップ
1. `.env.example`をコピーし`.env`を作成、APIキー・モデル指定
2. 必要パッケージをインストール
   ```
   pip install -r requirements.txt
   ```
3. `streamlit run app.py`で起動

## ディレクトリ
- `vector/` : ベクトルDB保存先
- `docs/`   : アップロード資料
