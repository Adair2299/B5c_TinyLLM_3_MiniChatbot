
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from model import TinyChatbot
from chat import generate

ROOT = Path(__file__).resolve().parent

app = FastAPI()

app.mount(
    "/static",
    StaticFiles(directory=ROOT / "static"),
    name="static"
)

# Load the trained Transformer model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load(
    ROOT / "model.pt",
    map_location=device,
    weights_only=True
)

vocabulary = checkpoint["vocabulary"]

model = TinyChatbot(**checkpoint["model_config"]).to(device)
model.load_state_dict(checkpoint["model_state"])
model.eval()

print(f"Model loaded successfully on {device}")


class ChatRequest(BaseModel):
    question: str


@app.get("/")
def home():
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat")
def chat(request: ChatRequest):
    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    answer = generate(
        model=model,
        vocabulary=vocabulary,
        question=question,
        device=device
    )

    return {"answer": answer}
