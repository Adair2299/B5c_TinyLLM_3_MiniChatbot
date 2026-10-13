
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from model import TinyChatbot
from chat import generate

import csv
import random

ROOT = Path(__file__).resolve().parent

# Load training examples from two datasets
dataset_pairs = []

for filename in ["dataset.csv", "dataset_2.csv"]:
    with (ROOT / filename).open(
        encoding="utf-8",
        newline=""
    ) as handle:
        reader = csv.DictReader(handle)

        dataset_pairs.extend(
            {
                "question": row["question"],
                "answer": row["answer"]
            }
            for row in reader
        )

print(f"Loaded {len(dataset_pairs)} training examples.")

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


# Random token API
@app.get("/api/random-tokens")
def random_tokens():
    if isinstance(vocabulary, dict):
        if "stoi" in vocabulary:
            tokens = list(vocabulary["stoi"].keys())
        elif "token_to_id" in vocabulary:
            tokens = list(vocabulary["token_to_id"].keys())
        elif "itos" in vocabulary:
            tokens = list(vocabulary["itos"])
        elif "id_to_token" in vocabulary:
            tokens = list(vocabulary["id_to_token"].values())
        else:
            tokens = list(vocabulary.keys())
    else:
        tokens = list(vocabulary)

    selected = random.sample(tokens, min(15, len(tokens)))

    return {"tokens": selected}


# Random training examples API
@app.get("/api/random-examples")
def random_examples():
    selected = random.sample(
        dataset_pairs,
        min(3, len(dataset_pairs))
    )

    return {"examples": selected}

