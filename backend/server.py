import os
import json
import asyncio
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from openai import AsyncOpenAI

# ─── Configuration ─────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
FRONTEND_DIR = str(BASE_DIR / "frontend")

# TODO: Update these with the URL and API Key provided by Beam after deployment
BEAM_URL = os.environ.get("BEAM_URL", "https://api.beam.cloud/v1") 
BEAM_API_KEY = os.environ.get("BEAM_API_KEY", "your-beam-auth-token")

# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(title="Qwen2.5 LoRA Chatbot (Beam Proxy)", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

@app.get("/")
async def serve_index():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

# ─── Request/Response schemas ─────────────────────────────────────────────────
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: list[Message]
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    system_prompt: str = "You are a helpful, friendly AI assistant."

@app.get("/health")
async def health():
    return {
        "status":  "ready",
        "backend": "Beam Cloud Proxy",
    }

async def stream_tokens(request: ChatRequest) -> AsyncGenerator[str, None]:
    client = AsyncOpenAI(
        base_url=BEAM_URL,
        api_key=BEAM_API_KEY,
    )

    conversation = [{"role": "system", "content": request.system_prompt}]
    for msg in request.messages:
        conversation.append({"role": msg.role, "content": msg.content})

    try:
        stream = await client.chat.completions.create(
            model="my_custom_model",
            messages=conversation,
            max_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stream=True
        )

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content is not None:
                token_text = chunk.choices[0].delta.content
                data = json.dumps({"token": token_text})
                yield f"data: {data}\n\n"
        
        yield "data: [DONE]\n\n"

    except Exception as e:
        data = json.dumps({"token": f"\n\n[Error communicating with Beam API: {str(e)}]"})
        yield f"data: {data}\n\n"
        yield "data: [DONE]\n\n"

@app.post("/chat")
async def chat(request: ChatRequest):
    return StreamingResponse(
        stream_tokens(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control":               "no-cache",
            "X-Accel-Buffering":           "no",
            "Access-Control-Allow-Origin": "*",
        },
    )
