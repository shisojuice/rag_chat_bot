import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from utils.embedding import get_embedding_model
from utils.model_selector import get_llm
import pickle

VECTOR_DIR = "vector"

class RAGChatBot:
    def __init__(self, model_name):
        load_dotenv()
        self.model_name = model_name
        self.llm = get_llm(model_name)
        self.embedding_model = get_embedding_model()
        self.vectorstore = self.load_vectorstore()

    def load_vectorstore(self):
        if not os.path.exists(VECTOR_DIR):
            os.makedirs(VECTOR_DIR)
        index_path = os.path.join(VECTOR_DIR, "faiss_index")
        if os.path.exists(index_path):
            return FAISS.load_local(index_path, self.embedding_model)
        return None

    def expand_query(self, query):
        """
        OpenAIでクエリのパラフレーズ・キーワード拡張を行う
        """
        try:
            expander = get_llm("openai")
            prompt = f"""
            次の質問を検索に適したキーワードや関連語をカンマ区切りで5つほど抽出してください。
            質問: {query}
            出力例: keyword1, keyword2, ...
            """
            keywords = expander(prompt)
            return query + ", " + keywords.replace("\n", " ")
        except Exception:
            return query

    def rerank(self, query, docs_and_scores):
        """
        OpenAIでqueryと各チャンクの類似度を再計算し、上位順にソート
        """
        try:
            scorer = get_llm("openai")
            results = []
            for doc, score in docs_and_scores:
                prompt = f"""
                質問: {query}\n---\n資料: {doc.page_content}\nこの資料は質問にどれだけ関連しますか？5段階で1(無関係)〜5(非常に関連)で数字のみ返答してください。\n"""
                res = scorer(prompt)
                try:
                    rank = int(res.strip()[0])
                except Exception:
                    rank = 3
                results.append((doc, score, rank))
            # rank降順、score昇順（距離が小さいほど近い）で再ソート
            results.sort(key=lambda x: (-x[2], x[1]))
            return [(doc, score) for doc, score, rank in results]
        except Exception:
            return docs_and_scores

    def ask(self, query):
        if self.vectorstore is None:
            return "ベクトルDBが未作成です。資料をアップロードしベクトル化してください。"
        # クエリ拡張
        expanded_query = self.expand_query(query)
        # 検索
        docs_and_scores = self.vectorstore.similarity_search_with_score(expanded_query, k=8)
        if not docs_and_scores:
            return "該当資料が見つかりませんでした。"
        # 再ランク
        docs_and_scores = self.rerank(query, docs_and_scores)[:5]
        # プロンプト設計強化
        context = "\n---\n".join([
            f"[{i+1}] {doc.page_content}\n(ファイル: {doc.metadata.get('source_file')} | フォルダパス: {doc.metadata.get('folder_path','')})"
            for i, (doc, _) in enumerate(docs_and_scores)
        ])
        prompt = f"""
あなたは社内資料のAIアシスタントです。以下の資料から根拠となる部分を明記し、質問に日本語で丁寧に答えてください。
必ず参考資料番号（[n]）および該当資料のフォルダパス（"フォルダパス:"で始まる部分）を回答内に含めてください。複数資料を要約・統合しても構いません。

質問: {query}

参考資料:
{context}
"""
        response = self.llm(prompt)
        return response
