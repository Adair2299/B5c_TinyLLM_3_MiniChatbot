import math
import random
from pathlib import Path

import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt

# ==========================================
# Configuration
# ==========================================

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model.pt"

# Embedding selection
RANDOM_SELECTION = False
NUM_RANDOM_TOKENS = 12

SELECTED_TOKENS = [
    "i", "you", "we", "like", "want",
    "code", "music", "weather", "sunny"
]

# Attention settings
INPUT_SENTENCE = "what do you want to do?"
LAYER_INDEX = 0
HEAD_INDEX = 0

# ==========================================
# Load checkpoint
# ==========================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu",
    weights_only=True
)

state = checkpoint["model_state"]
vocabulary = checkpoint["vocabulary"]

# Support common vocabulary storage formats
if isinstance(vocabulary, dict):
    if "stoi" in vocabulary:
        stoi = vocabulary["stoi"]
    elif "token_to_id" in vocabulary:
        stoi = vocabulary["token_to_id"]
    elif "itos" in vocabulary:
        stoi = {token: i for i, token in enumerate(vocabulary["itos"])}
    elif "id_to_token" in vocabulary:
        stoi = {
            token: int(i)
            for i, token in vocabulary["id_to_token"].items()
        }
    elif all(isinstance(v, int) for v in vocabulary.values()):
        stoi = vocabulary
    else:
        raise ValueError("Unsupported vocabulary format.")
else:
    stoi = {token: i for i, token in enumerate(vocabulary)}

# ==========================================
# Extract embedding matrix
# ==========================================

embedding_keys = [
    key for key, value in state.items()
    if "embedding.weight" in key
    and value.ndim == 2
    and value.shape[0] == len(stoi)
]

if not embedding_keys:
    raise KeyError("Token embedding weights not found.")

embedding_key = next(
    (key for key in embedding_keys if "token_embedding" in key),
    embedding_keys[0]
)

embedding = state[embedding_key].detach().float()

if RANDOM_SELECTION:
    tokens = random.sample(
        list(stoi.keys()),
        min(NUM_RANDOM_TOKENS, len(stoi))
    )
else:
    tokens = [token for token in SELECTED_TOKENS if token in stoi]

if not tokens:
    raise ValueError("No selected tokens exist in the vocabulary.")

token_ids = [stoi[token] for token in tokens]
selected_embeddings = embedding[token_ids].numpy()

# ==========================================
# Visualize token embeddings
# ==========================================

fig1, ax1 = plt.subplots(
    figsize=(13, max(4, len(tokens) * 0.45))
)

im1 = ax1.imshow(
    selected_embeddings,
    cmap="coolwarm",
    aspect="auto"
)

ax1.set_title("Trained Token Embedding Matrix")
ax1.set_xlabel("Embedding Dimension")
ax1.set_ylabel("Token")
ax1.set_yticks(range(len(tokens)))
ax1.set_yticklabels(tokens)

fig1.colorbar(im1, ax=ax1)
fig1.tight_layout()

# ==========================================
# Extract QKV projection weights
# ==========================================

prefix = f"transformer.layers.{LAYER_INDEX}.self_attn"

qkv_key = next(
    key for key in state
    if key.endswith(f"layers.{LAYER_INDEX}.self_attn.in_proj_weight")
)

prefix = qkv_key.rsplit(".in_proj_weight", 1)[0]
qkv_weight = state[qkv_key].detach().float()

qkv_bias = state.get(
    f"{prefix}.in_proj_bias",
    torch.zeros(qkv_weight.shape[0])
).detach().float()

d_model = embedding.shape[1]

W_Q, W_K, W_V = qkv_weight.chunk(3, dim=0)
b_Q, b_K, b_V = qkv_bias.chunk(3, dim=0)

# ==========================================
# Encode the input sentence
# ==========================================

# Simple word-level tokenizer for visualization.
# It assumes words and punctuation are separate tokens
# in the saved vocabulary.

import re

sentence_tokens = re.findall(
    r"[a-z]+|[?.!,]",
    INPUT_SENTENCE.lower()
)

missing = [token for token in sentence_tokens if token not in stoi]

if missing:
    raise ValueError(f"Tokens not found in vocabulary: {missing}")

input_ids = torch.tensor(
    [stoi[token] for token in sentence_tokens],
    dtype=torch.long
)

x = embedding[input_ids]

# Add positional embeddings if the model stores them
position_keys = [
    key for key in state
    if "position_embedding.weight" in key
    or "pos_embedding.weight" in key
]

if position_keys:
    position_embedding = state[position_keys[0]].detach().float()
    positions = torch.arange(len(input_ids))
    x = x + position_embedding[positions]

# ==========================================
# Compute attention weights
# ==========================================

# Note: This directly computes one attention projection
# from the embedding representation. For later layers,
# the actual input would require a full forward pass.

if LAYER_INDEX != 0:
    raise ValueError("Use LAYER_INDEX = 0 for this simplified visualization.")

Q = F.linear(x, W_Q, b_Q)
K = F.linear(x, W_K, b_K)

num_heads = checkpoint["model_config"].get("nhead", 4)
head_dim = d_model // num_heads

if not 0 <= HEAD_INDEX < num_heads:
    raise ValueError("HEAD_INDEX is out of range.")

Q = Q.reshape(len(sentence_tokens), num_heads, head_dim)
K = K.reshape(len(sentence_tokens), num_heads, head_dim)

q = Q[:, HEAD_INDEX, :]
k = K[:, HEAD_INDEX, :]

scores = q @ k.T / math.sqrt(head_dim)

causal_mask = torch.triu(
    torch.ones_like(scores, dtype=torch.bool),
    diagonal=1
)

scores = scores.masked_fill(causal_mask, float("-inf"))
attention = torch.softmax(scores, dim=-1).detach().numpy()

# ==========================================
# Visualize attention matrix
# ==========================================

fig2, ax2 = plt.subplots(figsize=(6, 5))

im2 = ax2.imshow(
    attention,
    cmap="Blues",
    vmin=0,
    vmax=1
)

ax2.set_title(
    f"Attention Matrix - Layer {LAYER_INDEX + 1}, "
    f"Head {HEAD_INDEX + 1}"
)

ax2.set_xlabel("Key Token")
ax2.set_ylabel("Query Token")

ax2.set_xticks(range(len(sentence_tokens)))
ax2.set_yticks(range(len(sentence_tokens)))

ax2.set_xticklabels(sentence_tokens, rotation=45, ha="right")
ax2.set_yticklabels(sentence_tokens)

fig2.colorbar(im2, ax=ax2)
fig2.tight_layout()

# ==========================================
# Show windows
# ==========================================

manager = plt.get_current_fig_manager()
manager.window.move(200, 100)
plt.show()
