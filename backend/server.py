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
    from gradio_client import Client

    # BEAM_URL should now point to the Hugging Face Space ID (e.g., tharuntej7373/qwen-zerogpu)
    client = Client(BEAM_URL, token=BEAM_API_KEY)

    conversation = [{"role": "system", "content": request.system_prompt}]
    for msg in request.messages:
        conversation.append({"role": msg.role, "content": msg.content})

    try:
        # Submit to the Gradio space
        job = client.submit(
            conversation,
            request.max_new_tokens,
            request.temperature,
            request.top_p,
            api_name="/chat"
        )

        previous_text = ""
        # iterate asynchronously (Gradio 4+ client supports async if we use threading, or we can just iterate.
        # Wait, job output is a synchronous generator. Since we are in an async def, 
        # doing blocking iteration is slightly suboptimal but works for low traffic, 
        # or we could use asyncio.to_thread. For simplicity, we just iterate.
        for partial_text in job:
            if partial_text.startswith(previous_text):
                token_text = partial_text[len(previous_text):]
            else:
                token_text = partial_text
            previous_text = partial_text
            
            data = json.dumps({"token": token_text})
            yield f"data: {data}\n\n"
            await asyncio.sleep(0.01) # yield control to event loop

        yield "data: [DONE]\n\n"

    except Exception as e:
        data = json.dumps({"token": f"\n\n[Error communicating with ZeroGPU Space: {str(e)}]"})
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
