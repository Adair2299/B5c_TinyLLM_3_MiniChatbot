"""Interactive English conversation with the trained tiny Transformer."""

from pathlib import Path

import torch

from data_utils import detokenize, encode
from model import TinyChatbot

import warnings

warnings.filterwarnings(
    "ignore",
    message=".*Torch was not compiled with flash attention.*"
)

ROOT = Path(__file__).parent


def generate(model, vocabulary, question, device, temperature=0.8, max_new_tokens=24):
    reverse_vocabulary = {index: word for word, index in vocabulary.items()}
    prompt = [vocabulary["<bos>"]] + encode(question, vocabulary) + [vocabulary["<sep>"]]
    if len(prompt) >= model.config["max_length"]:
        return "Your question is too long."
    tokens = prompt[:]
    model.eval()
    with torch.no_grad():
        for _ in range(min(max_new_tokens, model.config["max_length"] - len(tokens))):
            logits = model(torch.tensor([tokens], device=device))[0, -1] / temperature
            for special in ("<pad>", "<bos>", "<sep>", "<unk>"):
                logits[vocabulary[special]] = -float("inf")
            probabilities = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probabilities, 1).item()
            if next_id == vocabulary["<eos>"]:
                break
            tokens.append(next_id)
    answer = [reverse_vocabulary[index] for index in tokens[len(prompt):]]
    return detokenize(answer) if answer else "I do not know."


def main():
    path = ROOT / "model.pt"
    if not path.exists():
        print("No trained model found. Run python train.py first.")
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    vocabulary = checkpoint["vocabulary"]
    model = TinyChatbot(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    print("Tiny English Chatbot | Type 'exit' to quit.")
    while True:
        question = input("You: ").strip()
        if question.lower() in ("exit", "quit"):
            break
        if not question:
            continue
        print("AI:", generate(model, vocabulary, question, device))


if __name__ == "__main__":
    main()
