import re
from typing import List
from langchain.schema import Document

class SemanticTextSplitter:
    """
    セマンティック（意味単位）で文書を分割するシンプルな実装例。
    セクションタイトル、段落、文単位で分割。
    """
    def __init__(self, chunk_size=500, chunk_overlap=50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_documents(self, docs: List[Document]) -> List[Document]:
        chunked_docs = []
        for doc in docs:
            # セクションタイトルや改行で分割
            sections = re.split(r'\n{2,}|^# .+', doc.page_content, flags=re.MULTILINE)
            current_text = ""
            for section in sections:
                sentences = re.split(r'(?<=[。.!?])\s+', section)
                for sentence in sentences:
                    if not sentence.strip():
                        continue
                    if len(current_text) + len(sentence) > self.chunk_size:
                        chunked_docs.append(Document(page_content=current_text, metadata=doc.metadata.copy()))
                        # overlapを考慮
                        if self.chunk_overlap > 0 and len(current_text) > self.chunk_overlap:
                            current_text = current_text[-self.chunk_overlap:]
                        else:
                            current_text = ""
                    current_text += sentence
            if current_text:
                chunked_docs.append(Document(page_content=current_text, metadata=doc.metadata.copy()))
        return chunked_docs
