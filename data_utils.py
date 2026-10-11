"""Shared text preprocessing for training and chatting."""

import re

SPECIAL_TOKENS = ["<pad>", "<bos>", "<sep>", "<eos>", "<unk>"]


def tokenize(text):
    return re.findall(r"[a-z]+|[?.!,]", text.lower())


def build_vocabulary(pairs):
    words = set()
    for question, answer in pairs:
        words.update(tokenize(question))
        words.update(tokenize(answer))
    tokens = SPECIAL_TOKENS + sorted(words)
    return {token: index for index, token in enumerate(tokens)}


def encode(text, vocabulary):
    unknown_id = vocabulary["<unk>"]
    return [vocabulary.get(token, unknown_id) for token in tokenize(text)]


def detokenize(tokens):
    result = " ".join(tokens)
    result = re.sub(r"\s+([?.!,])", r"\1", result)
    return result[:1].upper() + result[1:]
