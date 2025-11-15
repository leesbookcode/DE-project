from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
import config

# 导入RAG功能
import rag

app = FastAPI()

# 原有纯GPT聊天请求模型（不变）
class ChatRequest(BaseModel):
    message: str
    model: str = config.DEFAULT_MODEL
    max_tokens: int = config.DEFAULT_MAX_TOKENS

# RAG聊天请求模型（支持检索参数）
class RAGChatRequest(BaseModel):
    message: str
    model: str = config.DEFAULT_MODEL
    max_tokens: int = config.DEFAULT_MAX_TOKENS
    top_k: int = 3  # 检索返回的文档数量

# 根路由
@app.get("/")
def read_root():
    return {"message": "RAG问答系统（支持TXT/PDF/Excel）", 
            "提示": f"请在 {config.KNOWLEDGE_BASE_PATH} 目录放入文档，重启服务后使用"}

# 原有纯GPT聊天接口（不变）
@app.post("/api/chat")
async def chat_with_gpt(request: ChatRequest):
    try:
        url = f"{config.OPENAI_BASE_URL}/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {config.OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": request.model,
            "messages": [{"role": "user", "content": request.message}],
            "max_tokens": request.max_tokens
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=headers, json=payload, timeout=30.0)
            response.raise_for_status()
            data = response.json()
        
        return {
            "success": True,
            "response": data["choices"][0]["message"]["content"],
            "model": request.model,
            "usage": data.get("usage", {})
        }
    except httpx.HTTPError as e:
        raise HTTPException(status_code=500, detail=f"HTTP error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# RAG相关逻辑（仅保留初始化和聊天接口）
@app.on_event("startup")
def startup_event():
    # 启动时自动加载知识库（TXT/PDF/Excel）
    rag.init_knowledge_base()

# RAG增强的聊天接口
@app.post("/api/rag/chat")
async def rag_chat(request: RAGChatRequest):
    try:
        # 1. 检索知识库
        context = rag.search_knowledge(request.message, top_k=request.top_k)
        
        # 2. 构造带上下文的提示词
        system_prompt = f"""
        请根据以下上下文回答用户问题。如果上下文没有相关信息，直接说"根据知识库无法回答"。
        上下文：
        {context}
        """
        
        # 3. 调用大模型
        url = f"{config.OPENAI_BASE_URL}/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {config.OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": request.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": request.message}
            ],
            "max_tokens": request.max_tokens
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=headers, json=payload, timeout=60.0)
            response.raise_for_status()
            data = response.json()
        
        return {
            "success": True,
            "response": data["choices"][0]["message"]["content"],
            "context": context,
            "model": request.model,
            "usage": data.get("usage", {})
        }
    except httpx.HTTPError as e:
        raise HTTPException(status_code=500, detail=f"API错误: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))