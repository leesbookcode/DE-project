from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
import config

app = FastAPI()

# Request model
class ChatRequest(BaseModel):
    message: str
    model: str = config.DEFAULT_MODEL
    max_tokens: int = config.DEFAULT_MAX_TOKENS

@app.get("/")
def read_root():
    return {"message": "Hello World"}

@app.post("/api/chat")
async def chat_with_gpt(request: ChatRequest):
    """Call GPT API using HTTP request"""
    try:
        # Prepare API request
        url = f"{config.OPENAI_BASE_URL}/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {config.OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": request.model,
            "messages": [
                {"role": "user", "content": request.message}
            ],
            "max_tokens": request.max_tokens
        }
        
        # Make HTTP request
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
