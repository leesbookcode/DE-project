# rag.py
import os
import pandas as pd  # 新增：处理Excel
from typing import List, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import (
    DirectoryLoader, TextLoader, PyPDFLoader
)
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
import config

# 全局向量数据库实例
vector_db: Optional[FAISS] = None


# 新增：自定义Excel加载器（将Excel表格转为文本）
class ExcelLoader:
    """加载Excel文件，将每个工作表内容转为文本"""
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self) -> List[Document]:
        documents = []
        try:
            # 读取Excel所有工作表
            xls = pd.ExcelFile(self.file_path)
            for sheet_name in xls.sheet_names:
                df = pd.read_excel(xls, sheet_name=sheet_name)
                # 将DataFrame转为文本（每行用逗号分隔，每行之间空行）
                text = "\n".join([", ".join(map(str, row)) for _, row in df.iterrows()])
                # 创建Document对象（包含文件名和工作表名）
                documents.append(Document(
                    page_content=text,
                    metadata={"source": self.file_path, "sheet": sheet_name}
                ))
        except Exception as e:
            print(f"Excel加载失败 {self.file_path}: {e}")
        return documents


def init_knowledge_base():
    """初始化知识库（启动时自动加载 TXT/PDF/Excel）"""
    global vector_db
    knowledge_path = config.KNOWLEDGE_BASE_PATH

    # 创建知识库目录（如果不存在）
    if not os.path.exists(knowledge_path):
        os.makedirs(knowledge_path)
        print(f"已创建知识库目录：{knowledge_path}，请放入TXT/PDF/Excel文档")
        return

    # 加载文档（支持 TXT/PDF/Excel）
    loaders = [
        # TXT文件
        DirectoryLoader(knowledge_path, glob="*.txt", loader_cls=TextLoader),
        # PDF文件
        DirectoryLoader(knowledge_path, glob="*.pdf", loader_cls=PyPDFLoader),
        # Excel文件（.xlsx/.xls）
        DirectoryLoader(
            knowledge_path,
            glob="*.xlsx",
            loader_cls=ExcelLoader  # 使用自定义Excel加载器
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
            print(f"文档加载失败: {e}")

    if not documents:
        print(f"警告：知识库目录 {knowledge_path} 中未找到任何文档（支持TXT/PDF/Excel）")
        return

    # 文档分块（统一处理所有格式的文本）
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
    )
    split_docs = text_splitter.split_documents(documents)
    print(f"知识库加载完成，共处理 {len(documents)} 个文档，分块为 {len(split_docs)} 段")

    # 创建向量库
    embeddings = OpenAIEmbeddings(
        openai_api_key=config.OPENAI_API_KEY,
        openai_api_base=config.OPENAI_BASE_URL
    )
    vector_db = FAISS.from_documents(split_docs, embeddings)


def search_knowledge(query: str, top_k: int = 3) -> str:
    """检索知识库中与问题相关的内容"""
    if not vector_db:
        return "知识库未初始化，请检查目录是否有文档"
    
    relevant_docs = vector_db.similarity_search(query, k=top_k)
    return "\n\n".join([doc.page_content for doc in relevant_docs])