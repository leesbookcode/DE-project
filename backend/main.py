from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
import config

import rag

app = FastAPI()

class ChatRequest(BaseModel):
    message: str
    model: str = config.DEFAULT_MODEL
    max_tokens: int = config.DEFAULT_MAX_TOKENS

class RAGChatRequest(BaseModel):
    message: str
    model: str = config.DEFAULT_MODEL
    max_tokens: int = config.DEFAULT_MAX_TOKENS
    top_k: int = 3

@app.get("/")
def read_root():
    return {"message": "RAG Q&A System (TXT/PDF/Excel)", 
            "note": f"Place documents in {config.KNOWLEDGE_BASE_PATH}, restart service to use"}

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

@app.on_event("startup")
def startup_event():
    rag.init_knowledge_base()

@app.post("/api/rag/chat")
async def rag_chat(request: RAGChatRequest):
    try:
        context = rag.search_knowledge(request.message, top_k=request.top_k)
        
        system_prompt = f"""
        Answer user questions based on the following context. If the context has no relevant information, say "Cannot answer based on knowledge base".
        Context:
        {context}
        """
        
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
        raise HTTPException(status_code=500, detail=f"API error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
