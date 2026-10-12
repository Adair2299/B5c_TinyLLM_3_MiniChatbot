import torch
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt

from pathlib import Path

# Load checkpoint
ROOT = Path(__file__).resolve().parent
checkpoint = torch.load(
    ROOT / "model.pt",
    map_location="cpu",
    weights_only=True
)

# Extract QKV weights from the first Transformer layer
state = checkpoint["model_state"]

qkv_key = next(
    key for key in state
    if "layers.0.self_attn.in_proj_weight" in key
)

W_Q, W_K, W_V = state[qkv_key].chunk(3, dim=0)

# Print matrix information
for name, W in zip(["Q", "K", "V"], [W_Q, W_K, W_V]):
    print(f"\n{name} Matrix | Shape: {W.shape}")
    print(W)

# Visualize matrices
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, name, W in zip(
    axes,
    ["Q", "K", "V"],
    [W_Q, W_K, W_V]
):
    im = ax.imshow(W.numpy(), cmap="coolwarm")
    ax.set_title(f"W_{name} Matrix")
    ax.set_xlabel("Input Dimension")
    ax.set_ylabel("Output Dimension")
    fig.colorbar(im, ax=ax)

manager = plt.get_current_fig_manager()
manager.window.move(200, 100)
plt.tight_layout()
plt.show()
