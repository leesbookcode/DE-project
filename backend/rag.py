import os
import pandas as pd
from typing import List, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import (
    DirectoryLoader, TextLoader, PyPDFLoader
)
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
import config

vector_db: Optional[FAISS] = None


class ExcelLoader:
    """Load Excel files and convert worksheet content to text"""
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self) -> List[Document]:
        documents = []
        try:
            xls = pd.ExcelFile(self.file_path)
            for sheet_name in xls.sheet_names:
                df = pd.read_excel(xls, sheet_name=sheet_name)
                text = "\n".join([", ".join(map(str, row)) for _, row in df.iterrows()])
                documents.append(Document(
                    page_content=text,
                    metadata={"source": self.file_path, "sheet": sheet_name}
                ))
        except Exception as e:
            print(f"Excel loading failed {self.file_path}: {e}")
        return documents


def init_knowledge_base():
    """Initialize knowledge base (auto-load TXT/PDF/Excel on startup)"""
    global vector_db
    knowledge_path = config.KNOWLEDGE_BASE_PATH

    if not os.path.exists(knowledge_path):
        os.makedirs(knowledge_path)
        print(f"Created knowledge base directory: {knowledge_path}, please add TXT/PDF/Excel documents")
        return

    loaders = [
        DirectoryLoader(knowledge_path, glob="*.txt", loader_cls=TextLoader),
        DirectoryLoader(knowledge_path, glob="*.pdf", loader_cls=PyPDFLoader),
        DirectoryLoader(
            knowledge_path,
            glob="*.xlsx",
            loader_cls=ExcelLoader
        ),
        DirectoryLoader(
            knowledge_path,
            glob="*.xls",
            loader_cls=ExcelLoader
        )
    ]

    documents: List[Document] = []
    for loader in loaders:
        try:
            documents.extend(loader.load())
        except Exception as e:
            print(f"Document loading failed: {e}")

    if not documents:
        print(f"Warning: No documents found in knowledge base directory {knowledge_path} (supports TXT/PDF/Excel)")
        return

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
    )
    split_docs = text_splitter.split_documents(documents)
    print(f"Knowledge base loaded: processed {len(documents)} documents, split into {len(split_docs)} chunks")

    embeddings = OpenAIEmbeddings(
        openai_api_key=config.OPENAI_API_KEY,
        openai_api_base=config.OPENAI_BASE_URL
    )
    vector_db = FAISS.from_documents(split_docs, embeddings)


def search_knowledge(query: str, top_k: int = 3) -> str:
    """Search knowledge base for content relevant to the query"""
    if not vector_db:
        return "Knowledge base not initialized, please check if documents exist in directory"
    
    relevant_docs = vector_db.similarity_search(query, k=top_k)
    return "\n\n".join([doc.page_content for doc in relevant_docs])
