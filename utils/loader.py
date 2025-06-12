import os
from langchain.document_loaders import PyPDFLoader, TextLoader, UnstructuredExcelLoader
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from utils.semantic_splitter import SemanticTextSplitter
import os
from langchain.vectorstores import FAISS
from utils.embedding import get_embedding_model

def save_and_embed_files(uploaded_files):
    VECTOR_DIR = os.environ["VECTOR_STORE_PATH"]
    DOCS_DIR = os.environ["DOCUMENT_PATH"]
    embedding_model = get_embedding_model()
    docs = []
    for uploaded_file in uploaded_files:
        # フォルダパスも取得（アップロード元のパス情報が含まれていれば）
        full_upload_name = uploaded_file.name
        folder_path = os.path.dirname(full_upload_name)
        filename = os.path.basename(full_upload_name)
        ext = filename.split('.')[-1].lower()
        save_path = os.path.join(DOCS_DIR, filename)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        if ext == "pdf":
            loader = PyPDFLoader(save_path)
        elif ext in ["xlsx", "xls"]:
            loader = UnstructuredExcelLoader(save_path)
        elif ext in ["txt", "md"]:
            loader = TextLoader(save_path, encoding="utf-8")
        elif ext in ["csv", "tsv"]:
            sep = "," if ext == "csv" else "\t"
            df = pd.read_csv(save_path, sep=sep, dtype=str)
            # 1行ずつDocument化（カラム名付き）
            from langchain.schema import Document
            for idx, row in df.iterrows():
                content = "\t".join([f"{col}: {row[col]}" for col in df.columns])
                docs.append(Document(page_content=content, metadata={"source": filename, "row": idx, "folder_path": folder_path}))
            continue
        else:
            continue
        docs.extend(loader.load())
    # チャンク分割＋メタデータ付与
    if os.getenv("SEMANTIC_SPLIT", "false").lower() == "true":
        splitter = SemanticTextSplitter(chunk_size=500, chunk_overlap=50)
    else:
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunked_docs = splitter.split_documents(docs)
    # メタデータ例: ファイル名、ページ番号等、フォルダパス
    for doc in chunked_docs:
        doc.metadata["source_file"] = doc.metadata.get("source", filename)
        doc.metadata["folder_path"] = folder_path
    # FAISSへ保存
    vectorstore = FAISS.from_documents(chunked_docs, embedding_model)
    index_path = os.path.join(VECTOR_DIR, "faiss_index")
    vectorstore.save_local(index_path)
